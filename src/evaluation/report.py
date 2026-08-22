"""Promotion gate and the final human-readable evaluation report.

The gate turns a pile of numbers into one of three verdicts:

* ``PASS``               - every hard requirement in configs/evaluation.yaml is met
* ``PASS_WITH_WARNINGS`` - no hard requirement failed, but something looks off
* ``FAIL``               - at least one hard requirement failed

Only ``PASS`` should be promoted / pushed to the Hub.
"""

from __future__ import annotations

from typing import Any

PASS = "PASS"
PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
FAIL = "FAIL"

DEFAULT_GATE = {
    "minimum_task_accuracy_improvement": 0.05,
    "minimum_generalization_improvement": 0.02,
    "maximum_allowed_forgetting": 0.10,
    "maximum_test_perplexity": 25.0,
    "maximum_leakage_rate": 0.0,
    "allow_overfitting": False,
}


def _check(
    name: str, passed: bool, actual: Any, required: Any, description: str
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "required": required,
        "description": description,
    }


def evaluate_gate(results: dict[str, Any], gate: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the promotion gate to an assembled results dict.

    ``results`` keys used (all optional - a missing input becomes a warning, never
    a silent pass):
    ``task`` {improvement.normalized_exact_match}, ``generalization``
    {primary_improvement}, ``forgetting`` {forgetting}, ``perplexity``
    {fine_tuned.test}, ``leakage`` {max_overlap_rate}, ``overfitting`` {status}.
    """
    gate = {**DEFAULT_GATE, **(gate or {})}
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    task = (results.get("task") or {}).get("improvement", {}).get("normalized_exact_match")
    if task is None:
        warnings.append("task accuracy improvement missing - gate check skipped")
    else:
        checks.append(
            _check(
                "task_accuracy_improvement",
                task >= gate["minimum_task_accuracy_improvement"],
                round(task, 4),
                f">= {gate['minimum_task_accuracy_improvement']}",
                "Fine-tuned minus base normalized exact match on the held-out test set.",
            )
        )

    gen = (results.get("generalization") or {}).get("primary_improvement")
    if gen is None:
        warnings.append("generalization improvement missing - gate check skipped")
    else:
        checks.append(
            _check(
                "generalization_improvement",
                gen >= gate["minimum_generalization_improvement"],
                round(gen, 4),
                f">= {gate['minimum_generalization_improvement']}",
                "Fine-tuned minus base token F1 on the reworded generalization set.",
            )
        )

    forgetting = (results.get("forgetting") or {}).get("forgetting")
    if forgetting is None:
        warnings.append("forgetting measurement missing - gate check skipped")
    else:
        checks.append(
            _check(
                "catastrophic_forgetting",
                forgetting <= gate["maximum_allowed_forgetting"],
                round(forgetting, 4),
                f"<= {gate['maximum_allowed_forgetting']}",
                "Base minus fine-tuned accuracy on the general-knowledge probe.",
            )
        )

    test_ppl = (results.get("perplexity") or {}).get("fine_tuned", {}).get("test")
    if test_ppl is None:
        warnings.append("test perplexity missing - gate check skipped")
    else:
        checks.append(
            _check(
                "test_perplexity",
                test_ppl <= gate["maximum_test_perplexity"],
                round(test_ppl, 4),
                f"<= {gate['maximum_test_perplexity']}",
                "Fine-tuned perplexity on the held-out test set.",
            )
        )

    leakage = (results.get("leakage") or {}).get("max_overlap_rate")
    if leakage is None:
        warnings.append("leakage rate missing - gate check skipped")
    else:
        checks.append(
            _check(
                "data_leakage",
                leakage <= gate["maximum_leakage_rate"],
                round(leakage, 4),
                f"<= {gate['maximum_leakage_rate']}",
                "Fraction of held-out questions that also appear in training.",
            )
        )

    overfitting_status = (results.get("overfitting") or {}).get("status")
    if overfitting_status is None:
        warnings.append("overfitting analysis missing - gate check skipped")
    elif not gate["allow_overfitting"]:
        checks.append(
            _check(
                "overfitting",
                overfitting_status not in ("strong_overfitting",),
                overfitting_status,
                "not strong_overfitting",
                "Loss-curve overfitting classification.",
            )
        )
        if overfitting_status == "possible_overfitting":
            warnings.append(
                "possible overfitting detected - consider fewer epochs or a lower LoRA rank"
            )

    underfitting_status = (results.get("underfitting") or {}).get("status")
    if underfitting_status == "possible_underfitting":
        warnings.append(
            "possible underfitting detected (heuristic) - consider more epochs or a higher LoRA rank"
        )

    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        verdict = FAIL
    elif warnings:
        verdict = PASS_WITH_WARNINGS
    else:
        verdict = PASS

    if not checks:
        verdict = FAIL
        warnings.append("no gate checks could be evaluated - evaluation results are incomplete")

    return {
        "verdict": verdict,
        "promote": verdict == PASS,
        "failed_checks": failed,
        "warnings": warnings,
        "checks": checks,
        "gate": gate,
        "explanation": _explain(verdict, failed, warnings),
    }


def _explain(verdict: str, failed: list[str], warnings: list[str]) -> str:
    if verdict == PASS:
        return "All hard requirements met with no warnings. The adapter is safe to promote."
    if verdict == FAIL:
        return (
            "Failed hard requirement(s): "
            + ", ".join(failed)
            + ". Do not promote this adapter; adjust the configuration and retrain."
        )
    return (
        "No hard requirement failed, but review these before promoting: "
        + "; ".join(warnings)
        + "."
    )


# ------------------------------------------------------------------- rendering


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "inf" if value == float("inf") else f"{value:.{digits}f}"
    return str(value)


def _describe_quantization(quant: dict[str, Any]) -> str:
    """Describe quantization honestly - a config with load_in_4bit off is not 4-bit."""
    if not quant:
        return "n/a"
    if not quant.get("load_in_4bit", True):
        return "disabled (full precision base weights)"
    return (
        f"4-bit {quant.get('quant_type', 'n/a')}, "
        f"double_quant={quant.get('double_quant', 'n/a')}"
    )


def _delta(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def render_final_report(results: dict[str, Any]) -> str:
    """Render artifacts/evaluation/final_report.md from the assembled results."""
    dataset = results.get("dataset", {})
    metadata = results.get("metadata", {})
    training = metadata.get("training", {})
    lora = metadata.get("lora", {})
    quant = metadata.get("quantization", {})
    params = metadata.get("parameters", {})
    overfitting = results.get("overfitting", {})
    underfitting = results.get("underfitting", {})
    perplexity = results.get("perplexity", {})
    task = results.get("task", {})
    generalization = results.get("generalization", {})
    forgetting = results.get("forgetting", {})
    leakage = results.get("leakage", {})
    gate = results.get("gate", {})

    lines: list[str] = ["# Model Fine-Tuning Evaluation Report", ""]
    lines += [
        f"- Run: `{metadata.get('run_name', 'unknown')}`",
        f"- Base model: `{metadata.get('model_name', 'unknown')}`",
        f"- Generated: {metadata.get('completed_utc', metadata.get('timestamp_utc', 'unknown'))}",
        f"- Git commit: `{metadata.get('git_commit') or 'not a git checkout'}`",
        f"- Seed: {metadata.get('seed', 'unknown')}",
        "",
        "## Dataset",
        "",
        "| Split | Examples |",
        "| --- | --- |",
    ]
    for key in ("total", "train", "validation", "test", "generalization", "general_knowledge"):
        if key in dataset:
            lines.append(f"| {key} | {dataset[key]} |")

    lines += [
        "",
        "## Training",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Epochs | {training.get('num_train_epochs', 'n/a')} |",
        f"| Learning rate | {training.get('learning_rate', 'n/a')} |",
        f"| Effective batch size | {metadata.get('effective_batch_size', 'n/a')} |",
        f"| Max sequence length | {training.get('max_seq_length', 'n/a')} |",
        f"| LoRA rank (r) | {lora.get('r', 'n/a')} |",
        f"| LoRA alpha | {lora.get('alpha', 'n/a')} |",
        f"| LoRA dropout | {lora.get('dropout', 'n/a')} |",
        f"| LoRA target modules | {', '.join(lora.get('resolved_target_modules') or []) or 'n/a'} |",
        f"| Quantization | {_describe_quantization(quant)} |",
        f"| Compute dtype | {metadata.get('compute_dtype', 'n/a')} |",
        f"| Trainable parameters | {params.get('trainable_parameters', 'n/a'):,} |"
        if isinstance(params.get("trainable_parameters"), int)
        else "| Trainable parameters | n/a |",
        f"| Total parameters | {params.get('total_parameters', 'n/a'):,} |"
        if isinstance(params.get("total_parameters"), int)
        else "| Total parameters | n/a |",
        f"| Trainable percentage | {params.get('trainable_percentage', 'n/a')}% |",
        "",
        "## Training Behavior",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Final training loss | {_fmt(overfitting.get('final_training_loss'))} |",
        f"| Best validation loss | {_fmt(overfitting.get('best_validation_loss'))} |",
        f"| Final validation loss | {_fmt(overfitting.get('final_validation_loss'))} |",
        f"| Generalization gap | {_fmt(overfitting.get('generalization_gap'))} |",
        f"| Best step | {overfitting.get('best_step', 'n/a')} |",
        f"| Overfitting status | **{overfitting.get('status', 'n/a')}** |",
        f"| Underfitting status | **{underfitting.get('status', 'n/a')}** (heuristic) |",
        "",
        "## Perplexity",
        "",
        "| Split | Base | Fine-tuned | Change |",
        "| --- | --- | --- | --- |",
    ]
    base_ppl = perplexity.get("base", {})
    ft_ppl = perplexity.get("fine_tuned", {})
    for split in ("train", "validation", "test"):
        b, f = base_ppl.get(split), ft_ppl.get(split)
        change = (f - b) if isinstance(b, (int, float)) and isinstance(f, (int, float)) else None
        lines.append(f"| {split} | {_fmt(b)} | {_fmt(f)} | {_delta(change)} |")
    lines.append("")
    lines.append("Lower is better. Only the test row is the headline number.")

    lines += ["", "## Task Performance (held-out test set)", "", _metric_table(task)]
    lines += ["", "## Generalization (reworded questions)", "", _metric_table(generalization)]
    audit = (generalization or {}).get("set_audit")
    if audit:
        lines.append("")
        lines.append(
            f"Set audit: max similarity to any training question "
            f"{_fmt(audit.get('max_similarity_to_train'))} "
            f"({'no near-copies' if audit.get('valid') else str(len(audit.get('too_similar', []))) + ' near-copies found'})."
        )

    lines += ["", "## Catastrophic Forgetting (general-knowledge probe)", ""]
    if forgetting:
        lines += [
            "| Metric | Base | Fine-tuned | Forgetting (base - fine-tuned) |",
            "| --- | --- | --- | --- |",
            f"| {forgetting.get('primary_metric', 'contains_reference')} | "
            f"{_fmt(forgetting.get('base', {}).get(forgetting.get('primary_metric', 'contains_reference')))} | "
            f"{_fmt(forgetting.get('fine_tuned', {}).get(forgetting.get('primary_metric', 'contains_reference')))} | "
            f"{_fmt(forgetting.get('forgetting'))} |",
            "",
            "Per category (positive = capability lost):",
            "",
            "| Category | Forgetting |",
            "| --- | --- |",
        ]
        for category, value in (forgetting.get("forgetting_by_category") or {}).items():
            lines.append(f"| {category} | {_delta(value)} |")
    else:
        lines.append("Not measured.")

    lines += ["", "## Leakage", ""]
    if leakage:
        lines += [
            "| Split | Records | Exact overlaps | Near overlaps | Overlap rate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, split in (leakage.get("splits") or {}).items():
            lines.append(
                f"| {name} | {split['records']} | {split['exact_overlaps']} | "
                f"{split['near_overlaps']} | {_fmt(split['overlap_rate'])} |"
            )
        lines.append("")
        lines.append(f"Max overlap rate: **{_fmt(leakage.get('max_overlap_rate'))}**")
    else:
        lines.append("Not measured.")

    lines += ["", "## Final Recommendation", "", f"### {gate.get('verdict', 'FAIL')}", ""]
    lines.append(gate.get("explanation", "No gate result available."))
    if gate.get("checks"):
        lines += [
            "",
            "| Check | Actual | Required | Result |",
            "| --- | --- | --- | --- |",
        ]
        for check in gate["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"| {check['name']} | {check['actual']} | {check['required']} | {mark} |"
            )
    if gate.get("warnings"):
        lines += ["", "**Warnings**", ""] + [f"- {w}" for w in gate["warnings"]]

    return "\n".join(lines) + "\n"


def _metric_table(comparison: dict[str, Any]) -> str:
    if not comparison:
        return "Not measured."
    base = comparison.get("base", {})
    finetuned = comparison.get("fine_tuned", {})
    improvement = comparison.get("improvement", {})
    rows = ["| Metric | Base | Fine-tuned | Improvement |", "| --- | --- | --- | --- |"]
    for metric in ("exact_match", "normalized_exact_match", "token_f1", "contains_reference"):
        if metric in base or metric in finetuned:
            rows.append(
                f"| {metric} | {_fmt(base.get(metric))} | {_fmt(finetuned.get(metric))} | "
                f"{_delta(improvement.get(metric))} |"
            )
    primary = comparison.get("primary_metric")
    if primary:
        rows += ["", f"Primary metric: `{primary}`."]
    return "\n".join(rows)
