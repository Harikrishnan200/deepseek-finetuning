"""Minimal FastAPI inference service.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Configured entirely through environment variables so no secrets or paths are
baked into the code:

    MODEL_NAME    base model id (default: the DeepSeek R1 distill)
    ADAPTER_PATH  LoRA adapter directory (default: artifacts/training/adapter)
    CONFIG_PATH   config used for the prompt template and quantization settings

The model is loaded lazily on the first /generate call so /health answers
immediately and the container starts fast.

Privacy: this service exposes *generated answers only*. It never reads or serves
data/raw/, data/processed/, or any evaluation predictions.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "artifacts/training/adapter")
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/qlora.yaml")

app = FastAPI(
    title="Personal QA - DeepSeek R1 Distill Qwen 1.5B + QLoRA",
    version="0.1.0",
    description="Answers questions about a single person's public profile.",
)

_generator: Any = None
_load_error: str | None = None
_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    response: str
    latency_seconds: float
    generated_tokens: int
    tokens_per_second: float


def get_generator() -> Any:
    """Load the model once, on first use. Thread-safe."""
    global _generator, _load_error
    if _generator is not None:
        return _generator
    with _lock:
        if _generator is not None:
            return _generator
        try:
            from src.inference.generate import Generator
            from src.training.config import load_config

            config = load_config(CONFIG_PATH)
            _generator = Generator.from_config(config, ADAPTER_PATH)
        except Exception as exc:  # surfaced through /health and /generate
            _load_error = f"{type(exc).__name__}: {exc}"
            raise
        return _generator


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "model_loaded": _generator is not None,
        "load_error": _load_error,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        generator = get_generator()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc

    try:
        result = generator.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    return GenerateResponse(**result.to_dict())
