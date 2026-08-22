"""Cross-split leakage detection.

Two questions are answered here:

1. Does any test/validation instruction appear *verbatim* in the training set?
2. Does any test/validation instruction appear as a *near duplicate* of a
   training instruction above the configured similarity threshold?

Both are computed locally with the same TF-IDF cosine model used for
deduplication. Findings are reported, never silently dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.deduplicate import DEFAULT_THRESHOLD, cross_similarity
from src.data.schema import Record, normalize_text


def exact_overlap(held_out: list[Record], train: list[Record]) -> list[dict[str, Any]]:
    """Held-out records whose normalized instruction also occurs in training."""
    train_index: dict[str, list[int]] = {}
    for record in train:
        train_index.setdefault(normalize_text(record.instruction), []).append(record.id)
    findings = []
    for record in held_out:
        key = normalize_text(record.instruction)
        if key in train_index:
            findings.append(
                {"held_out_id": record.id, "train_ids": train_index[key], "similarity": 1.0}
            )
    return findings


def near_overlap(
    held_out: list[Record],
    train: list[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    corpus: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Held-out records with a training instruction at similarity >= threshold.

    ``corpus`` is forwarded to :func:`cross_similarity`; pass every instruction in
    the dataset so these scores match the ones the splitter grouped on.
    """
    if not held_out or not train:
        return []
    sims = cross_similarity(
        [r.instruction for r in held_out], [r.instruction for r in train], corpus=corpus
    )
    findings = []
    for row, record in enumerate(held_out):
        best = int(np.argmax(sims[row]))
        score = float(sims[row, best])
        if score >= threshold:
            findings.append(
                {
                    "held_out_id": record.id,
                    "train_ids": [train[best].id],
                    "similarity": round(score, 4),
                }
            )
    return findings


def group_overlap(held_out: list[Record], train: list[Record]) -> list[int]:
    """Group ids present in both splits - should always be empty after splitting."""
    train_groups = {r.group_id for r in train if r.group_id >= 0}
    return sorted({r.group_id for r in held_out if r.group_id in train_groups})


def analyse_split_leakage(
    train: list[Record],
    validation: list[Record],
    test: list[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_examples: int = 100,
) -> dict[str, Any]:
    # Fit the TF-IDF weights once on every instruction in the dataset so the
    # similarity scale here is identical to the one used to build the groups.
    corpus = [r.instruction for r in (*train, *validation, *test)]
    result: dict[str, Any] = {
        "near_duplicate_threshold": threshold,
        "method": "normalized exact match + TF-IDF char n-gram cosine similarity",
        "splits": {},
    }
    worst_rate = 0.0
    for name, held_out in (("validation", validation), ("test", test)):
        exact = exact_overlap(held_out, train)
        near = near_overlap(held_out, train, threshold=threshold, corpus=corpus)
        shared_groups = group_overlap(held_out, train)
        near_ids = {f["held_out_id"] for f in near} | {f["held_out_id"] for f in exact}
        rate = len(near_ids) / len(held_out) if held_out else 0.0
        worst_rate = max(worst_rate, rate)
        result["splits"][name] = {
            "records": len(held_out),
            "exact_overlaps": len(exact),
            "near_overlaps": len(near),
            "overlapping_records": len(near_ids),
            "overlap_rate": round(rate, 4),
            "shared_group_ids": shared_groups,
            "exact_overlap_examples": exact[:max_examples],
            "near_overlap_examples": sorted(
                near, key=lambda f: -f["similarity"]
            )[:max_examples],
        }
    result["max_overlap_rate"] = round(worst_rate, 4)
    result["leakage_free"] = worst_rate == 0.0
    return result
