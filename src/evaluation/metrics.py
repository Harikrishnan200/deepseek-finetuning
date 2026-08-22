"""Task-specific metrics for personal-profile QA.

Perplexity alone does not tell you whether the model actually answers questions
about a person correctly, so this module adds string-overlap metrics. Everything
here is deterministic, local, and dependency-light - no LLM judge, no API, no cost.

All functions operate on plain strings, so they are unit-testable without torch.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from src.data.schema import normalize_text

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation/articles, collapse space."""
    tokens = [t for t in normalize_text(text).split() if t not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """Strict string equality after whitespace stripping only."""
    return float(prediction.strip() == reference.strip())


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Equality after case/punctuation/article normalization.

    This is the headline task-accuracy metric: it credits "N Harikrishnan." and
    "n harikrishnan" as the same answer, which is what we care about.
    """
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 - partial credit for answers that are right but verbose.

    The fine-tuned model often answers in a full sentence where the reference is
    terse (or vice versa), which exact match scores as a total failure. F1 is the
    metric to read when the gap between EM and F1 is large.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_reference(prediction: str, reference: str) -> float:
    """1.0 when the normalized reference appears inside the normalized prediction.

    Useful for the general-knowledge benchmark, where the gold answer is short
    ("4", "Paris") and any correct response should contain it.
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not ref:
        return 0.0
    return float(re.search(rf"(?<!\w){re.escape(ref)}(?!\w)", pred) is not None)


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    return {
        "exact_match": exact_match(prediction, reference),
        "normalized_exact_match": normalized_exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "contains_reference": contains_reference(prediction, reference),
    }


def score_predictions(
    predictions: Sequence[str], references: Sequence[str]
) -> list[dict[str, float]]:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must be the same length "
            f"({len(predictions)} != {len(references)})"
        )
    return [score_pair(p, r) for p, r in zip(predictions, references, strict=True)]


def aggregate_scores(scores: Iterable[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across examples. Empty input yields zeros, not NaN."""
    scores = list(scores)
    keys = ("exact_match", "normalized_exact_match", "token_f1", "contains_reference")
    if not scores:
        return dict.fromkeys(keys, 0.0)
    return {k: round(sum(s[k] for s in scores) / len(scores), 4) for k in keys}


def evaluate_predictions(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, Any]:
    per_example = score_predictions(predictions, references)
    return {"count": len(per_example), **aggregate_scores(per_example)}


def perplexity_from_loss(loss: float) -> float:
    """exp(mean cross-entropy). Clamped so a diverged run reports inf, not OverflowError."""
    if loss is None or math.isnan(loss):
        return float("nan")
    if loss > 20:
        return float("inf")
    return round(math.exp(loss), 4)
