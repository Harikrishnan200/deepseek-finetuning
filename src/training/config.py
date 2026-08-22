"""Typed loading and validation of the YAML experiment configs.

Importable without torch so that config validation runs in CPU-only CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_QUANT_TYPES = {"nf4", "fp4"}
VALID_COMPUTE_DTYPES = {"auto", "bfloat16", "float16", "float32"}

REQUIRED_TRAINING_KEYS = (
    "output_dir",
    "num_train_epochs",
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "max_seq_length",
    "logging_steps",
    "eval_steps",
    "save_steps",
)


class ConfigError(ValueError):
    """Raised when a config file is structurally invalid."""


@dataclass
class Config:
    """Parsed training configuration."""

    run_name: str
    model_name: str
    seed: int
    dataset: dict[str, str]
    prompt: dict[str, Any]
    split: dict[str, Any]
    training: dict[str, Any]
    lora: dict[str, Any]
    quantization: dict[str, Any]
    hub: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def effective_batch_size(self) -> int:
        return (
            self.training["per_device_train_batch_size"]
            * self.training["gradient_accumulation_steps"]
        )


def _require(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"{where} is missing required keys: {', '.join(missing)}")


def validate_config(data: dict[str, Any]) -> None:
    """Structural validation. Raises ConfigError with an actionable message."""
    _require(
        data,
        ("model_name", "seed", "dataset", "prompt", "split", "training", "lora", "quantization"),
        "config",
    )

    split = data["split"]
    _require(split, ("train_ratio", "validation_ratio", "test_ratio"), "split")
    total = split["train_ratio"] + split["validation_ratio"] + split["test_ratio"]
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"split ratios must sum to 1.0, got {total}")
    threshold = split.get("near_duplicate_threshold", 0.9)
    if not 0.0 < threshold <= 1.0:
        raise ConfigError(f"near_duplicate_threshold must be in (0, 1], got {threshold}")

    prompt = data["prompt"]
    _require(prompt, ("template", "response_marker"), "prompt")
    for placeholder in ("{instruction}", "{response}"):
        if placeholder not in prompt["template"]:
            raise ConfigError(f"prompt.template must contain {placeholder}")
    if prompt["response_marker"] not in prompt["template"]:
        raise ConfigError("prompt.response_marker must appear in prompt.template")

    _require(data["training"], REQUIRED_TRAINING_KEYS, "training")
    training = data["training"]
    for key in ("num_train_epochs", "learning_rate", "max_seq_length"):
        if not isinstance(training[key], (int, float)) or training[key] <= 0:
            raise ConfigError(f"training.{key} must be a positive number, got {training[key]!r}")
    for key in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    ):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ConfigError(f"training.{key} must be a positive integer, got {training[key]!r}")

    lora = data["lora"]
    _require(lora, ("r", "alpha", "dropout"), "lora")
    if not isinstance(lora["r"], int) or lora["r"] < 1:
        raise ConfigError(f"lora.r must be a positive integer, got {lora['r']!r}")
    if not 0.0 <= lora["dropout"] < 1.0:
        raise ConfigError(f"lora.dropout must be in [0, 1), got {lora['dropout']!r}")
    if not isinstance(lora.get("target_modules", []), list):
        raise ConfigError("lora.target_modules must be a list (empty means auto-detect)")

    quant = data["quantization"]
    _require(quant, ("load_in_4bit", "quant_type", "double_quant"), "quantization")
    if quant["quant_type"] not in VALID_QUANT_TYPES:
        raise ConfigError(
            f"quantization.quant_type must be one of {sorted(VALID_QUANT_TYPES)}, "
            f"got {quant['quant_type']!r}"
        )
    dtype = quant.get("compute_dtype", "auto")
    if dtype not in VALID_COMPUTE_DTYPES:
        raise ConfigError(
            f"quantization.compute_dtype must be one of {sorted(VALID_COMPUTE_DTYPES)}, got {dtype!r}"
        )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a YAML mapping, got {type(data).__name__}")
    validate_config(data)
    return Config(
        run_name=data.get("run_name", path.stem),
        model_name=data["model_name"],
        seed=int(data["seed"]),
        dataset=data["dataset"],
        prompt=data["prompt"],
        split=data["split"],
        training=data["training"],
        lora=data["lora"],
        quantization=data["quantization"],
        hub=data.get("hub", {}),
        source_path=str(path),
        raw=data,
    )


def load_evaluation_config(path: str | Path = "configs/evaluation.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"evaluation config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("evaluation config must be a YAML mapping")
    _require(data, ("datasets", "overfitting", "underfitting", "gate"), "evaluation config")
    _require(
        data["gate"],
        (
            "minimum_task_accuracy_improvement",
            "maximum_allowed_forgetting",
            "maximum_test_perplexity",
            "maximum_leakage_rate",
            "allow_overfitting",
        ),
        "gate",
    )
    return data
