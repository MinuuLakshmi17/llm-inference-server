import asyncio

import pytest

from app.scheduler import GenerationRequest, Scheduler


class FakeState:
    def __init__(self, logits, cache):
        self.logits = logits
        self.past_key_values = cache


class FakeRunner:
    eos_token_id = 999

    def __init__(self):
        self.prefill_calls = 0
        self.decode_calls = 0
        self.counter = 0

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_ids):
        return "".join(f"<{x}>" for x in token_ids)

    def prefill(self, input_ids):
        self.prefill_calls += 1
        return FakeState(0, {"length": len(input_ids)})

    def sample(self, logits, temperature, top_k):
        self.counter += 1
        return self.counter

    def decode_batch(self, token_ids, past_key_values, cache_lengths):
        self.decode_calls += 1
        return [0 for _ in token_ids], [{"length": n + 1} for n in cache_lengths]


@pytest.mark.asyncio
async def test_scheduler_generates_and_finishes():
    runner = FakeRunner()
    scheduler = Scheduler(runner, max_batch_size=2, max_queue_size=4)
    await scheduler.start()

    req = GenerationRequest("hello", max_new_tokens=3, temperature=0, top_k=1)
    await scheduler.submit(req)
    await scheduler.wait(req)

    assert req.done
    assert len(req.output_ids) == 3
    assert runner.prefill_calls == 1
    assert runner.decode_calls == 2

    await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_batches_active_requests():
    runner = FakeRunner()
    scheduler = Scheduler(runner, max_batch_size=2, max_queue_size=4)
    await scheduler.start()

    a = GenerationRequest("a", 4, 0, 1)
    b = GenerationRequest("b", 4, 0, 1)

    await scheduler.submit(a)
    await scheduler.submit(b)

    await asyncio.gather(scheduler.wait(a), scheduler.wait(b))

    assert len(a.output_ids) == 4
    assert len(b.output_ids) == 4
    assert runner.prefill_calls == 2

    await scheduler.stop()
