from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .config import Settings
from .scheduler import GenerationRequest, Scheduler


class InferenceService:
    def __init__(self, settings: Settings):
        from .models import ModelRunner
        self.settings = settings
        self.runner = ModelRunner(settings)
        self.scheduler = Scheduler(
            self.runner,
            max_batch_size=settings.max_batch_size,
            max_queue_size=settings.max_queue_size,
        )

    async def start(self):
        await self.scheduler.start()

    async def stop(self):
        await self.scheduler.stop()

    def create_request(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationRequest:
        return GenerationRequest(
            prompt=prompt,
            max_new_tokens=min(max_new_tokens, self.settings.max_new_tokens),
            temperature=temperature,
            top_k=top_k,
        )


def create_app(service: InferenceService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    service = service or InferenceService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(
        title="Continuous-Batching LLM Inference Server",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": settings.model_name,
            "device": str(service.runner.device),
            "active_sequences": len(service.scheduler.active),
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate")
    async def generate(payload: dict, request: Request):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must be a non-empty string")

        max_new_tokens = int(payload.get("max_new_tokens", 32))
        temperature = float(payload.get("temperature", 0.7))
        top_k = int(payload.get("top_k", 50))

        if max_new_tokens < 1:
            raise HTTPException(status_code=400, detail="max_new_tokens must be >= 1")
        if temperature < 0:
            raise HTTPException(status_code=400, detail="temperature must be >= 0")

        req = service.create_request(
            prompt,
            max_new_tokens,
            temperature,
            top_k,
        )

        try:
            await service.scheduler.submit(req)
            await asyncio.wait_for(
                service.scheduler.wait(req),
                timeout=settings.request_timeout_s,
            )
        except OverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="generation timed out")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        # The done event contains the complete response.
        event = None
        while True:
            item = await req.event_queue.get()
            if item["type"] == "done":
                event = item
                break
            if item["type"] == "error":
                raise HTTPException(status_code=500, detail=item["error"])

        return JSONResponse(event)

    @app.post("/generate/stream")
    async def generate_stream(payload: dict):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must be a non-empty string")

        max_new_tokens = int(payload.get("max_new_tokens", 32))
        temperature = float(payload.get("temperature", 0.7))
        top_k = int(payload.get("top_k", 50))

        req = service.create_request(prompt, max_new_tokens, temperature, top_k)
        try:
            await service.scheduler.submit(req)
        except OverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc))

        async def event_stream():
            async for event in service.scheduler.stream(req):
                yield f"event: {event['type']}\ndata: {__import__('json').dumps(event)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app
