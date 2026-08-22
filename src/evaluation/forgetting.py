"""Catastrophic-forgetting check.

Fine-tuning on ~620 narrowly-scoped personal facts can degrade the base model's
general ability. ``data/eval/general_knowledge.jsonl`` is a small, purely
general-purpose probe (basic maths, simple reasoning, common facts, Python,
general CS) containing no personal information. We score the base and fine-tuned
models on it and report the *drop*.

This is a smoke test, not a benchmark: ~36 items cannot certify that general
ability is intact, but a large drop is strong evidence that it is not.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from src.data.schema import Record
from src.evaluation.metrics import aggregate_scores, score_predictions

GenerateFn = Callable[[Sequence[str]], list[str]]

# Short gold answers ("43", "Paris"), so substring credit is the fair metric.
PRIMARY_METRIC = "contains_reference"


def evaluate_general_knowledge(
    records: list[Record], generate_fn: GenerateFn, *, label: str = "model"
) -> dict[str, Any]:
    predictions = list(generate_fn([r.instruction for r in records]))
    per_example = score_predictions(predictions, [r.response for r in records])

    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record, score in zip(records, per_example, strict=True):
        by_category[record.category or "uncategorised"].append(score)

    return {
        "label": label,
        "count": len(records),
        **aggregate_scores(per_example),
        "by_category": {
            category: {"count": len(scores), **aggregate_scores(scores)}
            for category, scores in sorted(by_category.items())
        },
        "primary_metric": PRIMARY_METRIC,
    }


def compare_forgetting(
    records: list[Record],
    base_generate: GenerateFn,
    finetuned_generate: GenerateFn,
    *,
    maximum_allowed_forgetting: float = 0.10,
) -> dict[str, Any]:
    """Report how much general capability the fine-tune cost.

    ``forgetting`` is positive when the fine-tuned model is *worse* than the base.
    """
    base = evaluate_general_knowledge(records, base_generate, label="base")
    finetuned = evaluate_general_knowledge(records, finetuned_generate, label="fine_tuned")

    forgetting = round(base[PRIMARY_METRIC] - finetuned[PRIMARY_METRIC], 4)
    per_category = {
        category: round(
            base["by_category"][category][PRIMARY_METRIC]
            - finetuned["by_category"].get(category, {}).get(PRIMARY_METRIC, 0.0),
            4,
        )
        for category in base["by_category"]
    }

    return {
        "dataset": "general_knowledge",
        "count": len(records),
        "primary_metric": PRIMARY_METRIC,
        "base": base,
        "fine_tuned": finetuned,
        "forgetting": forgetting,
        "forgetting_by_category": per_category,
        "maximum_allowed_forgetting": maximum_allowed_forgetting,
        "within_tolerance": forgetting <= maximum_allowed_forgetting,
        "note": (
            "Small probe set - a large drop is meaningful evidence of forgetting, "
            "but a small drop is not proof that general ability is intact."
        ),
    }
