"""Publish the LoRA adapter and evaluation artifacts to the Hugging Face Hub.

Privacy rules enforced here:

* the token comes from the ``HF_TOKEN`` environment variable only - never a file,
  never an argument that would land in shell history;
* ``data/raw/`` and ``data/processed/`` are **never** uploaded;
* only the adapter weights, tokenizer, model card, and the aggregate evaluation
  reports go to the Hub. Aggregate reports contain scores, not answers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Evaluation artifacts that are safe to publish: aggregate numbers only.
PUBLISHABLE_ARTIFACTS = (
    "final_report.md",
    "evaluation.json",
    "base_vs_finetuned.json",
    "perplexity.json",
    "generalization.json",
    "forgetting_report.json",
    "overfitting_report.json",
    "underfitting_report.json",
    "final_leakage_report.json",
    "gate.json",
)
# Never publish these, even if someone points --evaluation-dir at them.
BLOCKED_NAMES = ("test_predictions.json", "personal_dataset.jsonl")


def get_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Export it (see .env.example) or add it as a Kaggle secret. "
            "Never hard-code it."
        )
    return token


def build_model_card(
    model_id: str, config: Any, metadata: dict[str, Any], results: dict[str, Any] | None
) -> str:
    params = (metadata or {}).get("parameters", {})
    lora = (metadata or {}).get("lora", {})
    gate = (results or {}).get("gate", {})
    task = (results or {}).get("task", {})

    lines = [
        "---",
        "library_name: peft",
        f"base_model: {config.model_name}",
        "tags:",
        "  - peft",
        "  - lora",
        "  - qlora",
        "  - question-answering",
        "license: mit",
        "---",
        "",
        f"# {model_id}",
        "",
        f"QLoRA adapter for `{config.model_name}`, fine-tuned to answer questions about one "
        "person's public profile (education, projects, skills, interests).",
        "",
        "This is **an adapter, not a full model**. Load it on top of the base model with PEFT.",
        "",
        "## Intended use and limitations",
        "",
        "- Answers questions about a single individual's non-sensitive profile information.",
        "- It is **not** a general assistant. Fine-tuning on a narrow dataset can degrade "
        "general capability; see the catastrophic-forgetting section of the evaluation report.",
        "- Answers may be confidently wrong. Do not use it as an authoritative source.",
        "- The training dataset itself is **not** published.",
        "",
        "## Training",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        "| Method | QLoRA (4-bit NF4 + double quantization) + SFT |",
        f"| LoRA rank / alpha / dropout | {lora.get('r')} / {lora.get('alpha')} / {lora.get('dropout')} |",
        f"| Target modules | {', '.join(lora.get('resolved_target_modules') or []) or 'n/a'} |",
        f"| Trainable parameters | {params.get('trainable_parameters', 'n/a')} "
        f"({params.get('trainable_percentage', 'n/a')}% of total) |",
        f"| Epochs | {(metadata or {}).get('training', {}).get('num_train_epochs', 'n/a')} |",
        f"| Learning rate | {(metadata or {}).get('training', {}).get('learning_rate', 'n/a')} |",
        f"| Seed | {(metadata or {}).get('seed', 'n/a')} |",
        "",
        "## Prompt format",
        "",
        "```",
        config.prompt["template"].replace("{instruction}", "<your question>").replace("{response}", ""),
        "```",
        "",
    ]

    if task:
        lines += [
            "## Evaluation (held-out test set)",
            "",
            "| Metric | Base | Fine-tuned |",
            "| --- | --- | --- |",
        ]
        for metric in ("normalized_exact_match", "token_f1"):
            lines.append(
                f"| {metric} | {task.get('base', {}).get(metric, 'n/a')} | "
                f"{task.get('fine_tuned', {}).get(metric, 'n/a')} |"
            )
        lines.append("")
    if gate:
        lines += [f"**Promotion gate verdict: {gate.get('verdict')}** - {gate.get('explanation')}", ""]

    lines += [
        "## Usage",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'base_id = "{config.model_name}"',
        f'adapter_id = "{model_id}"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(adapter_id)",
        "model = AutoModelForCausalLM.from_pretrained(base_id, device_map='auto')",
        "model = PeftModel.from_pretrained(model, adapter_id)",
        "```",
        "",
        "Produced by a fully free/open-source pipeline: Kaggle free GPU for training, "
        "GitHub Actions for CPU validation, Hugging Face Hub for distribution.",
    ]
    return "\n".join(lines) + "\n"


def push_adapter(
    adapter_dir: str | Path,
    model_id: str,
    config: Any,
    *,
    private: bool = True,
    metadata: dict[str, Any] | None = None,
    evaluation_dir: str | Path = "artifacts/evaluation",
    require_pass: bool = False,
) -> str:
    """Upload the adapter, tokenizer, model card, and safe evaluation artifacts."""
    from huggingface_hub import HfApi

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")

    evaluation_dir = Path(evaluation_dir)
    results = None
    results_file = evaluation_dir / "evaluation.json"
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))

    if require_pass:
        verdict = ((results or {}).get("gate") or {}).get("verdict")
        if verdict != "PASS":
            raise RuntimeError(
                f"refusing to publish: promotion gate verdict is {verdict or 'missing'}, not PASS"
            )

    token = get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=model_id, private=private, exist_ok=True)

    card = build_model_card(model_id, config, metadata or {}, results)
    (adapter_dir / "README.md").write_text(card, encoding="utf-8")

    # Adapter weights + tokenizer + card.
    api.upload_folder(
        repo_id=model_id,
        folder_path=str(adapter_dir),
        commit_message="Upload QLoRA adapter, tokenizer and model card",
        ignore_patterns=["*.lock", "__pycache__/*", *BLOCKED_NAMES],
    )

    # Aggregate evaluation artifacts only.
    for name in PUBLISHABLE_ARTIFACTS:
        path = evaluation_dir / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"evaluation/{name}",
                repo_id=model_id,
                commit_message=f"Add evaluation artifact {name}",
            )
    plots_dir = evaluation_dir / "plots"
    if plots_dir.exists():
        api.upload_folder(
            repo_id=model_id,
            folder_path=str(plots_dir),
            path_in_repo="evaluation/plots",
            commit_message="Add evaluation plots",
        )

    return f"https://huggingface.co/{model_id}"
