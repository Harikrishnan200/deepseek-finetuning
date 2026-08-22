"""Exact and near-duplicate detection.

Near duplicates are found with a local TF-IDF character n-gram model and cosine
similarity - no embedding API, no network, no cost. The similarity threshold is
configurable and nothing is ever deleted automatically: this module only
*detects and reports*.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.schema import Record, normalize_text

DEFAULT_THRESHOLD = 0.90


@dataclass(frozen=True)
class SimilarPair:
    left_id: int
    right_id: int
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "similarity": round(self.similarity, 4),
        }


def exact_duplicate_groups(records: Iterable[Record], *, field: str = "pair") -> list[list[int]]:
    """Group record ids that are exactly identical after normalization.

    ``field`` is one of ``"pair"`` (instruction+response), ``"instruction"``, or
    ``"response"``. Only groups of size >= 2 are returned.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if field == "pair":
            key = f"{normalize_text(record.instruction)}\x1f{normalize_text(record.response)}"
        elif field == "instruction":
            key = normalize_text(record.instruction)
        elif field == "response":
            key = normalize_text(record.response)
        else:  # pragma: no cover - programmer error
            raise ValueError(f"unknown field: {field}")
        buckets[key].append(record.id)
    return [sorted(ids) for ids in buckets.values() if len(ids) > 1]


def build_vectorizer() -> TfidfVectorizer:
    """Character 3-5 gram TF-IDF: robust to word order and small rewordings."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )


def similarity_matrix(texts: list[str]) -> np.ndarray:
    """Dense cosine-similarity matrix over normalized texts."""
    if len(texts) < 2:
        return np.zeros((len(texts), len(texts)), dtype=np.float32)
    matrix = build_vectorizer().fit_transform([normalize_text(t) for t in texts])
    # TfidfVectorizer L2-normalizes rows, so the Gram matrix is cosine similarity.
    return np.asarray((matrix @ matrix.T).todense(), dtype=np.float32)


def near_duplicate_pairs(
    records: list[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    field: str = "instruction",
) -> list[SimilarPair]:
    """All record pairs whose ``field`` similarity is >= ``threshold``."""
    if len(records) < 2:
        return []
    texts = [getattr(r, field) for r in records]
    sims = similarity_matrix(texts)
    rows, cols = np.triu_indices(len(records), k=1)
    mask = sims[rows, cols] >= threshold
    return [
        SimilarPair(records[int(i)].id, records[int(j)].id, float(sims[int(i), int(j)]))
        for i, j in zip(rows[mask], cols[mask], strict=True)
    ]


def cross_similarity(
    left_texts: list[str],
    right_texts: list[str],
    *,
    corpus: list[str] | None = None,
) -> np.ndarray:
    """Cosine similarity of every left text against every right text.

    ``corpus`` controls what the TF-IDF vocabulary and IDF weights are fitted on.
    Pass the *whole* dataset here whenever the result must agree with
    :func:`similarity_matrix` - IDF is corpus-dependent, so fitting on a subset
    shifts every score slightly and would make grouping and leakage detection
    disagree about which pairs sit above the threshold. Defaults to the union of
    the two sides.
    """
    if not left_texts or not right_texts:
        return np.zeros((len(left_texts), len(right_texts)), dtype=np.float32)
    vectorizer = build_vectorizer()
    fit_texts = corpus if corpus is not None else left_texts + right_texts
    vectorizer.fit([normalize_text(t) for t in fit_texts])
    left = vectorizer.transform([normalize_text(t) for t in left_texts])
    right = vectorizer.transform([normalize_text(t) for t in right_texts])
    return np.asarray((left @ right.T).todense(), dtype=np.float32)


def build_duplicate_report(
    records: list[Record], *, threshold: float = DEFAULT_THRESHOLD, max_examples: int = 200
) -> dict[str, Any]:
    exact_pairs = exact_duplicate_groups(records, field="pair")
    exact_instructions = exact_duplicate_groups(records, field="instruction")
    near_instruction = near_duplicate_pairs(records, threshold=threshold, field="instruction")
    near_response = near_duplicate_pairs(records, threshold=threshold, field="response")

    exact_ids = {i for group in exact_pairs for i in group[1:]}
    return {
        "near_duplicate_threshold": threshold,
        "method": "TF-IDF char_wb 3-5 grams, cosine similarity (local, no API)",
        "policy": "detect and report only - no record is deleted automatically",
        "total_records": len(records),
        "exact_duplicate_pair_groups": len(exact_pairs),
        "exact_duplicate_redundant_records": len(exact_ids),
        "exact_duplicate_instruction_groups": len(exact_instructions),
        "near_duplicate_instruction_pairs": len(near_instruction),
        "near_duplicate_response_pairs": len(near_response),
        "exact_duplicate_groups": exact_pairs[:max_examples],
        "exact_duplicate_instruction_id_groups": exact_instructions[:max_examples],
        "near_duplicate_instruction_examples": [
            p.to_dict()
            for p in sorted(near_instruction, key=lambda p: -p.similarity)[:max_examples]
        ],
    }
