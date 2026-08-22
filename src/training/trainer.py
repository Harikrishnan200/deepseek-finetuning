"""QLoRA supervised fine-tuning with TRL's SFTTrainer.

Design note on API stability: instead of relying on TRL's internal dataset
preparation (whose argument names have churned across releases), this module
tokenizes the dataset itself, masks the prompt tokens out of the labels, and
hands SFTTrainer a pre-tokenized dataset with ``skip_prepare_dataset``. The
SFTConfig is built by filtering our settings against the signature actually
installed, so a newer/older TRL does not break the run.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainerCallback

from src.data.schema import Record, load_records
from src.training.config import Config
from src.training.model import (
    count_parameters,
    load_base_model,
    load_tokenizer,
    prepare_peft_model,
    resolve_compute_dtype,
)

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- data


def build_example(
    record: Record,
    tokenizer: Any,
    prompt: dict[str, Any],
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize one record into input_ids / attention_mask / labels.

    When ``train_on_completion_only`` is set, every token belonging to the prompt
    (everything up to and including the response marker) is set to -100 so the
    loss is computed on the answer alone. That matters here: the instructions are
    templated and repetitive, and training on them would mostly teach the model to
    reproduce question phrasing.
    """
    template: str = prompt["template"]
    marker: str = prompt["response_marker"]

    full_text = template.format(instruction=record.instruction, response=record.response)
    if prompt.get("append_eos", True):
        full_text += tokenizer.eos_token

    marker_end = template.format(instruction=record.instruction, response="").index(marker) + len(marker)
    prompt_text = template.format(instruction=record.instruction, response="")[:marker_end]

    encoded = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    labels = list(input_ids)

    if prompt.get("train_on_completion_only", True):
        prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        prompt_length = min(prompt_length, len(labels))
        labels[:prompt_length] = [IGNORE_INDEX] * prompt_length
        if all(label == IGNORE_INDEX for label in labels):
            # Truncation ate the whole answer; keep the example trainable rather
            # than feeding the optimizer an all-masked row (which yields NaN loss).
            labels = list(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def build_dataset(
    records: list[Record], tokenizer: Any, prompt: dict[str, Any], max_seq_length: int
) -> Dataset:
    rows = [build_example(r, tokenizer, prompt, max_seq_length) for r in records]
    return Dataset.from_list(rows)


@dataclass
class PaddingCollator:
    """Pad a batch to its longest sequence; label padding uses -100."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = longest - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ------------------------------------------------------------------- bookkeeping


class HistoryCallback(TrainerCallback):
    """Collect every logged train/eval point into a single tidy history."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4)}
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 2:
            self.history.append(entry)


def merge_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the train-log and eval-log entries emitted at the same step."""
    by_step: dict[int, dict[str, Any]] = {}
    for entry in history:
        by_step.setdefault(entry["step"], {"step": entry["step"]}).update(entry)
    merged = [by_step[s] for s in sorted(by_step)]
    for entry in merged:
        if "loss" in entry:
            entry["train_loss"] = entry.pop("loss")
        if "eval_loss" in entry:
            entry["validation_loss"] = entry.pop("eval_loss")
    return merged


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "not installed"
    return versions


def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_run_metadata(config: Config, param_summary: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to reproduce this run."""
    return {
        "run_name": config.run_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "model_name": config.model_name,
        "seed": config.seed,
        "config_path": config.source_path,
        "dataset_hashes": {
            name: file_sha256(path) for name, path in config.dataset.items()
        },
        "training": config.training,
        "lora": {**config.lora, "resolved_target_modules": param_summary.get("target_modules")},
        "quantization": config.quantization,
        "prompt": config.prompt,
        "parameters": {
            k: param_summary[k]
            for k in ("trainable_parameters", "total_parameters", "trainable_percentage")
            if k in param_summary
        },
        "effective_batch_size": config.effective_batch_size,
        "compute_dtype": str(resolve_compute_dtype(config.quantization.get("compute_dtype", "auto"))),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "library_versions": _library_versions(),
    }


# ---------------------------------------------------------------------- training


def build_sft_config(config: Config, output_dir: Path) -> Any:
    """Build an SFTConfig, keeping only keys the installed TRL actually accepts."""
    from trl import SFTConfig

    training = config.training
    desired: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "run_name": config.run_name,
        "seed": config.seed,
        "num_train_epochs": float(training["num_train_epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "optim": training.get("optim", "paged_adamw_8bit"),
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "per_device_eval_batch_size": training["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "warmup_ratio": training.get("warmup_ratio", 0.05),
        "weight_decay": training.get("weight_decay", 0.01),
        "max_grad_norm": training.get("max_grad_norm", 0.3),
        "logging_steps": training["logging_steps"],
        "eval_strategy": "steps",
        "eval_steps": training["eval_steps"],
        "save_strategy": "steps",
        "save_steps": training["save_steps"],
        "save_total_limit": training.get("save_total_limit", 2),
        "load_best_model_at_end": training.get("load_best_model_at_end", True),
        "metric_for_best_model": training.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": False,
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": [],  # no W&B / no paid tracking
        "save_safetensors": True,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "max_length": training["max_seq_length"],
        "max_seq_length": training["max_seq_length"],
    }
    dtype = resolve_compute_dtype(config.quantization.get("compute_dtype", "auto"))
    desired["bf16"] = dtype == torch.bfloat16
    desired["fp16"] = dtype == torch.float16

    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    # SFTConfig inherits most fields from TrainingArguments via dataclass fields.
    accepted |= {f.name for f in getattr(SFTConfig, "__dataclass_fields__", {}).values()}
    filtered = {k: v for k, v in desired.items() if k in accepted}
    dropped = sorted(set(desired) - set(filtered))
    if dropped:
        print(f"[config] TRL {_library_versions()['trl']} does not accept: {dropped} (ignored)")
    return SFTConfig(**filtered)


def train(config: Config, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run QLoRA SFT end to end and write the training artifacts."""
    from transformers import set_seed
    from trl import SFTTrainer

    set_seed(config.seed)
    output_dir = Path(output_dir or config.training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.model_name)
    train_records, _ = load_records(config.dataset["train"])
    val_records, _ = load_records(config.dataset["validation"])
    print(f"[data] train={len(train_records)} validation={len(val_records)}")

    max_seq_length = config.training["max_seq_length"]
    train_dataset = build_dataset(train_records, tokenizer, config.prompt, max_seq_length)
    eval_dataset = build_dataset(val_records, tokenizer, config.prompt, max_seq_length)

    base_model = load_base_model(config.model_name, config.quantization, for_training=True)
    print(f"[model] loaded {type(base_model).__name__}")
    model, param_summary = prepare_peft_model(
        base_model,
        config.lora,
        gradient_checkpointing=config.training.get("gradient_checkpointing", True),
    )
    print(f"[lora] target modules: {param_summary['target_modules']}")
    print(
        f"[lora] trainable {param_summary['trainable_parameters']:,} / "
        f"{param_summary['total_parameters']:,} "
        f"({param_summary['trainable_percentage']}%)"
    )

    history_callback = HistoryCallback()
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(config, output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PaddingCollator(tokenizer.pad_token_id),
        callbacks=[history_callback],
    )

    metadata = build_run_metadata(config, param_summary)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Archive the exact config next to the run, so an experiment directory is
    # self-describing without needing the repo state it was launched from.
    if config.source_path and Path(config.source_path).exists():
        (output_dir / "config.yaml").write_text(
            Path(config.source_path).read_text(encoding="utf-8"), encoding="utf-8"
        )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    history = merge_history(history_callback.history)
    (output_dir / "training_history.json").write_text(
        json.dumps({"run_name": config.run_name, "history": history}, indent=2), encoding="utf-8"
    )

    metadata["parameters"] = count_parameters(trainer.model)
    metadata["completed_utc"] = datetime.now(UTC).isoformat()
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "adapter_path": str(adapter_dir),
        "history": history,
        "metadata": metadata,
        "parameter_summary": param_summary,
    }
