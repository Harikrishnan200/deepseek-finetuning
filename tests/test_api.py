"""FastAPI service and Hugging Face publishing - the parts that need no GPU."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.inference.types import GenerationResult
from src.publish import BLOCKED_NAMES, PUBLISHABLE_ARTIFACTS, build_model_card, get_token
from src.training.config import load_config


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(api, "_generator", None)
    monkeypatch.setattr(api, "_load_error", None)
    return TestClient(api.app)


class StubGenerator:
    """Stands in for the real model so the API contract is testable on CPU."""

    def __init__(self, response: str = "N Harikrishnan.") -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(self, prompt, *, max_new_tokens=256, temperature=0.0):
        self.calls.append(
            {"prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": temperature}
        )
        return GenerationResult(self.response, latency_seconds=0.5, generated_tokens=10)


def test_app_starts_and_health_responds(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False  # lazy load: /health must not pull the model


def test_generate_returns_response_and_metadata(client, monkeypatch):
    stub = StubGenerator()
    monkeypatch.setattr(api, "get_generator", lambda: stub)

    response = client.post("/generate", json={"prompt": "What is his full name?"})
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "N Harikrishnan."
    assert body["generated_tokens"] == 10
    assert body["latency_seconds"] == pytest.approx(0.5)
    assert body["tokens_per_second"] == pytest.approx(20.0)


def test_generate_forwards_parameters(client, monkeypatch):
    stub = StubGenerator()
    monkeypatch.setattr(api, "get_generator", lambda: stub)
    client.post("/generate", json={"prompt": "hi", "max_new_tokens": 32, "temperature": 0.7})
    assert stub.calls[0]["max_new_tokens"] == 32
    assert stub.calls[0]["temperature"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "payload",
    [
        {},                                        # missing prompt
        {"prompt": ""},                            # empty prompt
        {"prompt": "hi", "max_new_tokens": 0},     # out of range
        {"prompt": "hi", "temperature": 5.0},      # out of range
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/generate", json=payload).status_code == 422


def test_model_load_failure_returns_503(client, monkeypatch):
    def boom():
        raise RuntimeError("no weights here")

    monkeypatch.setattr(api, "get_generator", boom)
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 503
    assert "model unavailable" in response.json()["detail"]


def test_generation_failure_returns_500(client, monkeypatch):
    class Broken(StubGenerator):
        def generate(self, *args, **kwargs):
            raise RuntimeError("cuda oom")

    monkeypatch.setattr(api, "get_generator", lambda: Broken())
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 500


# ------------------------------------------------------------------ publishing


def test_get_token_requires_the_environment_variable(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
        get_token()
    monkeypatch.setenv("HF_TOKEN", "hf_example")
    assert get_token() == "hf_example"


def test_model_card_has_frontmatter_and_limitations():
    config = load_config("configs/qlora.yaml")
    metadata = {
        "parameters": {"trainable_parameters": 18_464_768, "trainable_percentage": 1.03},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "resolved_target_modules": ["q_proj"]},
        "training": {"num_train_epochs": 5, "learning_rate": 2e-4},
        "seed": 42,
    }
    card = build_model_card("user/model", config, metadata, None)
    assert card.startswith("---\nlibrary_name: peft")
    assert f"base_model: {config.model_name}" in card
    assert "## Intended use and limitations" in card
    assert "not** a general assistant" in card
    assert "training dataset itself is **not** published" in card


def test_publish_allowlist_holds_only_aggregate_reports():
    """Anything carrying raw personal answers must never be in the allowlist."""
    assert "test_predictions.json" not in PUBLISHABLE_ARTIFACTS
    assert "personal_dataset.jsonl" not in PUBLISHABLE_ARTIFACTS
    assert "test_predictions.json" in BLOCKED_NAMES
    assert "personal_dataset.jsonl" in BLOCKED_NAMES
    assert not any(name.endswith(".jsonl") for name in PUBLISHABLE_ARTIFACTS)


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("Harikrishnan200/deepseek-personal-qlora", True),   # Hub repo id
        ("owner/name", True),
        ("configs", True),                                   # existing local dir
        ("no-such-directory", False),
        ("./adapter", False),                                # explicit relative path
        ("/tmp/definitely/missing", False),
        ("a/b/c", False),                                    # too deep to be a Hub id
        ("owner/", False),
    ],
)
def test_adapter_reference_accepts_hub_ids_and_local_dirs(reference, expected):
    """A Hub id must not be mistaken for a missing path - that silently serves the base model."""
    from src.inference.generate import is_adapter_reference

    assert is_adapter_reference(reference) is expected


def test_resolve_model_id_passes_real_namespaces_through(monkeypatch):
    """A genuine owner/name must not trigger a network call."""
    import src.publish as publish

    monkeypatch.setattr(publish, "whoami", lambda: pytest.fail("should not call the Hub"))
    assert publish.resolve_model_id("someuser/my-adapter") == "someuser/my-adapter"


@pytest.mark.parametrize("given", ["YOUR_USERNAME/my-adapter", "your-real-hf-username/my-adapter"])
def test_resolve_model_id_replaces_placeholder_namespaces(monkeypatch, given):
    """Placeholder namespaces caused 403s; substitute the token's real owner."""
    import src.publish as publish

    monkeypatch.setattr(publish, "whoami", lambda: "realuser")
    assert publish.resolve_model_id(given) == "realuser/my-adapter"


def test_resolve_model_id_fills_in_a_missing_namespace(monkeypatch):
    import src.publish as publish

    monkeypatch.setattr(publish, "whoami", lambda: "realuser")
    assert publish.resolve_model_id(None) == "realuser/deepseek-personal-qlora"
    assert publish.resolve_model_id("custom-name") == "realuser/custom-name"
