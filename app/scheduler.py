from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .metrics import (
    ACTIVE_SEQUENCES,
    BATCH_SIZE,
    GENERATED_TOKENS,
    GENERATION_SECONDS,
    KV_SEQUENCES,
    QUEUE_TIME,
    REQUESTS,
    REQUEST_LATENCY,
    TTFT,
)

logger = logging.getLogger(__name__)


class RunnerProtocol(Protocol):
    eos_token_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def prefill(self, input_ids: list[int]) -> Any: ...
    def decode_one(self, token_id: int, past_key_values: Any, cache_length: int): ...
    def sample(self, logits, temperature: float, top_k: int) -> int: ...


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    created_at: float = field(default_factory=time.perf_counter)
    admitted_at: Optional[float] = None
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None

    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    past_key_values: Any = None
    logits: Any = None
    done: bool = False
    error: Optional[str] = None

    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)

    def remaining(self) -> int:
        return self.max_new_tokens - len(self.output_ids)


class Scheduler:
    """Single-owner event-loop scheduler.

    One scheduler loop owns all mutable request state. Model execution is
    synchronous and therefore intentionally serialized within a scheduling
    step. A future CUDA implementation can replace the runner while keeping
    the lifecycle and scheduling contract.
    """

    def __init__(self, runner: RunnerProtocol, max_batch_size: int, max_queue_size: int):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size

        self.queue: asyncio.Queue[GenerationRequest] = asyncio.Queue(maxsize=max_queue_size)
        self.active: list[GenerationRequest] = []
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    async def start(self):
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="continuous-batch-scheduler")

    async def stop(self):
        self._stopping = True
        if self._task:
            await self._task

    async def submit(self, request: GenerationRequest):
        if self._stopping:
            raise RuntimeError("Scheduler is stopping.")
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull:
            raise OverflowError("Inference queue is full.")
        self._wakeup.set()
        await self.start()

    async def wait(self, request: GenerationRequest):
        await request.finished_event.wait()
        if request.error:
            raise RuntimeError(request.error)
        return request

    async def stream(self, request: GenerationRequest):
        while True:
            event = await request.event_queue.get()
            yield event
            if event["type"] in ("done", "error"):
                break

    def _admit_waiting(self):
        while len(self.active) < self.max_batch_size and not self.queue.empty():
            request = self.queue.get_nowait()
            self.queue.task_done()
            request.admitted_at = time.perf_counter()
            QUEUE_TIME.observe(request.admitted_at - request.created_at)

            try:
                request.input_ids = self.runner.encode(request.prompt)
                if not request.input_ids:
                    raise ValueError("Prompt produced zero tokens.")

                # Prefill is done independently because prompts can have different lengths.
                state = self.runner.prefill(request.input_ids)
                request.past_key_values = state.past_key_values
                request.logits = state.logits
                self.active.append(request)
                KV_SEQUENCES.set(len(self.active))
                ACTIVE_SEQUENCES.set(len(self.active))
            except Exception as exc:
                self._fail(request, str(exc))

    async def _run(self):
        while not self._stopping:
            self._admit_waiting()

            if not self.active:
                # Idle: wait for new submissions. NOTE: queue.join() cannot
                # be used here. task_done() accounting means join() returns
                # immediately when the queue is drained, and awaiting an
                # already-completable coroutine never yields to the event
                # loop -- the result was a synchronous busy-spin that
                # starved the loop and deadlocked the server on startup.
                # An explicit wakeup event (set by submit()) is the correct
                # primitive; the timeout bounds the worst-case admission
                # latency if a wakeup races with clear().
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
                continue

            step_start = time.perf_counter()
            BATCH_SIZE.observe(len(self.active))

            completed: list[GenerationRequest] = []

            step_tokens = []
            decode_requests = []

            # Sample independently from each sequence's current logits.
            for request in list(self.active):
                try:
                    token = self.runner.sample(request.logits, request.temperature, request.top_k)
                    step_tokens.append(token)
                    decode_requests.append(request)
                    request.output_ids.append(token)
                    now = time.perf_counter()
                    if request.first_token_at is None:
                        request.first_token_at = now
                        TTFT.observe(now - (request.admitted_at or now))
                    await request.event_queue.put({"type": "token", "request_id": request.request_id, "token_id": token, "token": self.runner.decode([token])})
                    GENERATED_TOKENS.inc()
                    if token == self.runner.eos_token_id or not request.remaining():
                        completed.append(request)
                except Exception as exc:
                    self._fail(request, str(exc))
                    completed.append(request)

            # CRITICAL: all still-active sequences enter ONE transformer forward pass.
            decode_requests = [r for r in decode_requests if r not in completed and not r.done]
            if decode_requests:
                try:
                    logits, caches = self.runner.decode_batch(
                        [r.output_ids[-1] for r in decode_requests],
                        [r.past_key_values for r in decode_requests],
                        [len(r.input_ids) + len(r.output_ids) - 1 for r in decode_requests],
                    )
                    for i, request in enumerate(decode_requests):
                        request.logits = logits[i:i+1]
                        request.past_key_values = caches[i]
                except Exception as exc:
                    for request in decode_requests:
                        self._fail(request, str(exc))
                        if request not in completed:
                            completed.append(request)

            elapsed = time.perf_counter() - step_start
            GENERATION_SECONDS.inc(elapsed)

            for request in completed:
                self._finish(request)

            # Yield control after every scheduler step so HTTP clients and
            # newly-arriving requests get CPU time.
            await asyncio.sleep(0)

    def _finish(self, request: GenerationRequest):
        if request.done:
            return
        request.done = True
        request.finished_at = time.perf_counter()

        latency = request.finished_at - request.created_at
        REQUEST_LATENCY.observe(latency)
        REQUESTS.labels(status="success").inc()

        request.event_queue.put_nowait({
            "type": "done",
            "request_id": request.request_id,
            "text": self.runner.decode(request.output_ids),
            "input_tokens": len(request.input_ids),
            "output_tokens": len(request.output_ids),
            "latency_ms": latency * 1000,
            "time_to_first_token_ms": (
                (request.first_token_at - request.created_at) * 1000
                if request.first_token_at
                else None
            ),
        })
        request.finished_event.set()

        if request in self.active:
            self.active.remove(request)

        KV_SEQUENCES.set(len(self.active))
        ACTIVE_SEQUENCES.set(len(self.active))

    def _fail(self, request: GenerationRequest, message: str):
        if request.done:
            return

        request.error = message
        request.done = True
        request.finished_at = time.perf_counter()
        REQUESTS.labels(status="error").inc()

        try:
            request.event_queue.put_nowait({
                "type": "error",
                "request_id": request.request_id,
                "error": message,
            })
        finally:
            request.finished_event.set()

        if request in self.active:
            self.active.remove(request)

        KV_SEQUENCES.set(len(self.active))
        ACTIVE_SEQUENCES.set(len(self.active))
