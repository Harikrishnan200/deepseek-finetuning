"""Full evaluation: perplexity, task metrics, generalization, forgetting, gate.

Everything here compares the **base** and **fine-tuned** models on exactly the
same inputs, using the same decoding settings, so the differences reported are
attributable to the adapter and not to the harness.

The test split is only ever touched from this module, after training is finished.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from src.data.leakage import analyse_split_leakage
from src.data.prepare import write_json
from src.data.schema import Record, load_records
from src.evaluation.forgetting import compare_forgetting
from src.evaluation.generalization import compare_generalization
from src.evaluation.metrics import aggregate_scores, perplexity_from_loss, score_predictions
from src.evaluation.overfitting import analyse_overfitting, analyse_underfitting
from src.evaluation.report import evaluate_gate, render_final_report
from src.training.config import Config
from src.training.model import load_base_model, load_finetuned_model, load_tokenizer
from src.training.trainer import IGNORE_INDEX, build_example

# ------------------------------------------------------------------ perplexity


@torch.no_grad()
def compute_perplexity(
    model: Any,
    tokenizer: Any,
    records: list[Record],
    prompt: dict[str, Any],
    max_seq_length: int,
    *,
    batch_size: int = 4,
) -> dict[str, float]:
    """Token-weighted mean cross-entropy over the response tokens, and its exp().

    Losses are weighted by token count rather than averaged per batch, so the
    number does not depend on how the records happen to be batched.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        examples = [build_example(r, tokenizer, prompt, max_seq_length) for r in chunk]
        longest = max(len(e["input_ids"]) for e in examples)
        input_ids, attention_mask, labels = [], [], []
        for example in examples:
            pad = longest - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention_mask.append(example["attention_mask"] + [0] * pad)
            labels.append(example["labels"] + [IGNORE_INDEX] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
        }
        label_tensor = torch.tensor(labels, device=device)
        logits = model(**batch).logits.float()

        # Standard causal shift: predict token t+1 from position t.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = label_tensor[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_tokens = int((shift_labels != IGNORE_INDEX).sum())
        total_loss += float(loss)
        total_tokens += n_tokens

    mean_loss = total_loss / total_tokens if total_tokens else float("nan")
    return {
        "loss": round(mean_loss, 6),
        "perplexity": perplexity_from_loss(mean_loss),
        "tokens": total_tokens,
        "examples": len(records),
    }


# ------------------------------------------------------------------ generation


def make_generate_fn(
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    *,
    max_new_tokens: int = 128,
    batch_size: int = 4,
):
    """Return a deterministic batched ``generate_fn(instructions) -> responses``.

    Greedy decoding (do_sample=False) so base and fine-tuned runs are comparable
    and the whole evaluation is reproducible.
    """
    template: str = prompt["template"]
    model.eval()

    def generate(instructions: Sequence[str]) -> list[str]:
        outputs: list[str] = []
        original_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # required for correct batched generation
        try:
            device = next(model.parameters()).device
            for start in range(0, len(instructions), batch_size):
                chunk = list(instructions[start : start + batch_size])
                texts = [template.format(instruction=i, response="") for i in chunk]
                encoded = tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[:, encoded["input_ids"].shape[1] :]
                outputs.extend(
                    text.strip()
                    for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                )
        finally:
            tokenizer.padding_side = original_side
        return outputs

    return generate


def evaluate_task(
    records: list[Record], generate_fn, *, label: str = "model"
) -> tuple[dict[str, Any], list[str]]:
    predictions = list(generate_fn([r.instruction for r in records]))
    per_example = score_predictions(predictions, [r.response for r in records])
    return (
        {"label": label, "count": len(records), **aggregate_scores(per_example)},
        predictions,
    )


# ------------------------------------------------------------------- pipeline


def run_full_evaluation(
    config: Config,
    eval_config: dict[str, Any],
    adapter_path: str | Path,
    *,
    output_dir: str | Path = "artifacts/evaluation",
    training_history_path: str | Path = "artifacts/training/training_history.json",
    run_metadata_path: str | Path = "artifacts/training/run_metadata.json",
    save_predictions: bool = False,
) -> dict[str, Any]:
    """Run every evaluation stage and write all artifacts. Returns the results dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = config.prompt
    max_seq_length = config.training["max_seq_length"]
    batch_size = eval_config.get("batch_size", 4)
    max_new_tokens = eval_config.get("max_new_tokens", 128)

    paths = eval_config["datasets"]
    train_records, _ = load_records(paths["train"])
    val_records, _ = load_records(paths["validation"])
    test_records, _ = load_records(paths["test"])
    gen_records, _ = load_records(paths["generalization"])
    gk_records, _ = load_records(paths["general_knowledge"])

    results: dict[str, Any] = {
        "dataset": {
            "total": len(train_records) + len(val_records) + len(test_records),
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records),
            "generalization": len(gen_records),
            "general_knowledge": len(gk_records),
        }
    }

    # --- reproducibility metadata + training-curve analysis -------------------
    metadata = _read_json(run_metadata_path, default={})
    results["metadata"] = metadata
    history = (_read_json(training_history_path, default={}) or {}).get("history", [])
    results["overfitting"] = analyse_overfitting(history, eval_config.get("overfitting"))
    results["underfitting"] = analyse_underfitting(history, eval_config.get("underfitting"))
    write_json(output_dir / "overfitting_report.json", results["overfitting"])
    write_json(output_dir / "underfitting_report.json", results["underfitting"])

    # --- final leakage verification, before any test number is produced ------
    threshold = config.split["near_duplicate_threshold"]
    results["leakage"] = analyse_split_leakage(
        train_records, val_records, test_records, threshold=threshold
    )
    write_json(output_dir / "final_leakage_report.json", results["leakage"])
    print(f"[leakage] max overlap rate: {results['leakage']['max_overlap_rate']}")

    tokenizer = load_tokenizer(config.model_name)
    perplexity: dict[str, Any] = {"base": {}, "fine_tuned": {}}
    task_scores: dict[str, Any] = {}
    generalization_fns: dict[str, Any] = {}
    forgetting_fns: dict[str, Any] = {}
    predictions_dump: dict[str, list[str]] = {}

    # Base and fine-tuned are loaded one at a time: a Kaggle T4 has 16 GB and
    # holding two copies plus generation KV cache is a real OOM risk.
    for label in ("base", "fine_tuned"):
        print(f"[eval] loading {label} model ...")
        if label == "base":
            model = load_base_model(config.model_name, config.quantization, for_training=False)
        else:
            model = load_finetuned_model(config.model_name, str(adapter_path), config.quantization)

        for split_name, split_records in (
            ("train", train_records),
            ("validation", val_records),
            ("test", test_records),
        ):
            stats = compute_perplexity(
                model, tokenizer, split_records, prompt, max_seq_length, batch_size=batch_size
            )
            perplexity[label][split_name] = stats["perplexity"]
            perplexity[label][f"{split_name}_loss"] = stats["loss"]
            print(f"[eval] {label} {split_name} ppl={stats['perplexity']}")

        generate_fn = make_generate_fn(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, batch_size=batch_size
        )
        scores, preds = evaluate_task(test_records, generate_fn, label=label)
        task_scores[label] = scores
        if save_predictions:
            predictions_dump[label] = preds
        print(f"[eval] {label} test normalized EM={scores['normalized_exact_match']}")

        # Cache the generated answers so the comparison functions below do not
        # need both models resident at once.
        generalization_fns[label] = _replay(generate_fn([r.instruction for r in gen_records]))
        forgetting_fns[label] = _replay(generate_fn([r.instruction for r in gk_records]))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = ("exact_match", "normalized_exact_match", "token_f1", "contains_reference")
    results["perplexity"] = perplexity
    results["task"] = {
        "dataset": "test",
        "count": len(test_records),
        "base": task_scores["base"],
        "fine_tuned": task_scores["fine_tuned"],
        "improvement": {
            m: round(task_scores["fine_tuned"][m] - task_scores["base"][m], 4) for m in metrics
        },
        "primary_metric": "normalized_exact_match",
    }
    results["generalization"] = compare_generalization(
        gen_records,
        generalization_fns["base"],
        generalization_fns["fine_tuned"],
        train=train_records,
        threshold=threshold,
    )
    results["forgetting"] = compare_forgetting(
        gk_records,
        forgetting_fns["base"],
        forgetting_fns["fine_tuned"],
        maximum_allowed_forgetting=eval_config["gate"]["maximum_allowed_forgetting"],
    )
    results["gate"] = evaluate_gate(results, eval_config["gate"])

    write_json(output_dir / "perplexity.json", results["perplexity"])
    write_json(output_dir / "base_vs_finetuned.json", {
        "perplexity": results["perplexity"],
        "task": results["task"],
        "generalization": results["generalization"],
        "forgetting": results["forgetting"],
    })
    write_json(output_dir / "generalization.json", results["generalization"])
    write_json(output_dir / "forgetting_report.json", results["forgetting"])
    write_json(output_dir / "gate.json", results["gate"])
    write_json(output_dir / "evaluation.json", results)
    (output_dir / "final_report.md").write_text(render_final_report(results), encoding="utf-8")

    if save_predictions:
        # Contains personal answers - opt-in only, and gitignored by default.
        write_json(
            output_dir / "test_predictions.json",
            {
                "warning": "contains model output about a real person - do not publish",
                "instructions": [r.instruction for r in test_records],
                "references": [r.response for r in test_records],
                **predictions_dump,
            },
        )

    print(f"\n[gate] verdict: {results['gate']['verdict']}")
    return results


def _replay(outputs: list[str]):
    """Wrap already-generated outputs as a generate_fn, so comparisons stay pure."""

    def generate(_instructions: Sequence[str]) -> list[str]:
        return list(outputs)

    return generate


def _read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        print(f"[warn] missing {path} - continuing without it")
        return default
    return json.loads(path.read_text(encoding="utf-8"))
