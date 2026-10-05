import pytest
from fastapi.testclient import TestClient

from app.scheduler import Scheduler
from app.server import InferenceService, create_app
from app.config import Settings


class FakeState:
    def __init__(self, logits, cache):
        self.logits = logits
        self.past_key_values = cache


class FakeRunner:
    eos_token_id = 999
    device = "cpu"

    def encode(self, text):
        return [1, 2]

    def decode(self, token_ids):
        return "ok"

    def prefill(self, input_ids):
        return FakeState(0, {"length": len(input_ids)})

    def sample(self, logits, temperature, top_k):
        return 1

    def decode_batch(self, token_ids, past_key_values, cache_lengths):
        return [0 for _ in token_ids], [{"length": n + 1} for n in cache_lengths]


@pytest.fixture
def client():
    settings = Settings(
        model_name="fake",
        device="cpu",
        max_batch_size=4,
        max_queue_size=8,
        max_new_tokens=8,
        request_timeout_s=5,
        model_dtype="float32",
        log_level="INFO",
    )
    service = InferenceService.__new__(InferenceService)
    service.settings = settings
    service.runner = FakeRunner()
    service.scheduler = Scheduler(service.runner, 4, 8)

    app = create_app(service=service, settings=settings)

    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate(client):
    response = client.post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 2, "temperature": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output_tokens"] == 2
    assert "text" in body
