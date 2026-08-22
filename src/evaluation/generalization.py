"""Generalization evaluation: same knowledge, different wording.

``data/eval/generalization.jsonl`` is hand-curated: each question asks about a
fact that appears in the *training* split, but phrased differently enough that
answering it correctly requires having learned the fact rather than memorised the
question string. ``audit_generalization_set`` enforces that property.

The evaluator takes a ``generate_fn`` callable rather than a model, so the scoring
logic is unit-testable on CPU with a stub.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.data.deduplicate import DEFAULT_THRESHOLD, cross_similarity
from src.data.schema import Record
from src.evaluation.metrics import aggregate_scores, score_predictions

GenerateFn = Callable[[Sequence[str]], list[str]]


def audit_generalization_set(
    generalization: list[Record],
    train: list[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Check that no generalization question is a near copy of a training question.

    If this fails, the generalization score is measuring memorisation instead.
    """
    if not generalization or not train:
        return {"checked": 0, "too_similar": [], "valid": True}
    sims = cross_similarity(
        [r.instruction for r in generalization],
        [r.instruction for r in train],
        corpus=[r.instruction for r in (*generalization, *train)],
    )
    too_similar = []
    for row, record in enumerate(generalization):
        best = int(sims[row].argmax())
        score = float(sims[row, best])
        if score >= threshold:
            too_similar.append(
                {"generalization_id": record.id, "train_id": train[best].id, "similarity": round(score, 4)}
            )
    return {
        "checked": len(generalization),
        "threshold": threshold,
        "max_similarity_to_train": round(float(sims.max(axis=1).max()), 4),
        "mean_similarity_to_train": round(float(sims.max(axis=1).mean()), 4),
        "too_similar": too_similar,
        "valid": not too_similar,
    }


def evaluate_generalization(
    records: list[Record],
    generate_fn: GenerateFn,
    *,
    label: str = "model",
) -> dict[str, Any]:
    """Score one model on the generalization set.

    Token F1 and contains_reference are the metrics to read here: the reference
    answers are short atomic facts while models tend to reply in full sentences,
    so strict exact match under-reports real performance.
    """
    predictions = list(generate_fn([r.instruction for r in records]))
    per_example = score_predictions(predictions, [r.response for r in records])
    return {
        "label": label,
        "count": len(records),
        **aggregate_scores(per_example),
        "primary_metric": "token_f1",
    }


def compare_generalization(
    records: list[Record],
    base_generate: GenerateFn,
    finetuned_generate: GenerateFn,
    *,
    train: list[Record] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    base = evaluate_generalization(records, base_generate, label="base")
    finetuned = evaluate_generalization(records, finetuned_generate, label="fine_tuned")
    metrics = ("exact_match", "normalized_exact_match", "token_f1", "contains_reference")
    result = {
        "dataset": "generalization",
        "count": len(records),
        "base": base,
        "fine_tuned": finetuned,
        "improvement": {m: round(finetuned[m] - base[m], 4) for m in metrics},
        "primary_metric": "token_f1",
        "primary_improvement": round(finetuned["token_f1"] - base["token_f1"], 4),
    }
    if train is not None:
        result["set_audit"] = audit_generalization_set(records, train, threshold=threshold)
    return result
