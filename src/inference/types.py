"""Torch-free inference value types, so the API contract is testable on CPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    response: str
    latency_seconds: float
    generated_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return (
            round(self.generated_tokens / self.latency_seconds, 2) if self.latency_seconds else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "latency_seconds": round(self.latency_seconds, 4),
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": self.tokens_per_second,
        }
