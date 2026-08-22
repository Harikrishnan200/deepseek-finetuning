"""Leakage-aware train/validation/test splitting.

Records are first grouped so that semantically-equivalent questions land in the
same split, then whole *groups* are assigned to splits. Splitting individual
records would let a reworded variant of a training question show up in the test
set, which would make the test score measure memorisation instead of
generalisation.

The assignment is greedy over groups sorted largest-first, with a seeded shuffle
to break ties, so it is fully deterministic for a given seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

from src.data.deduplicate import DEFAULT_THRESHOLD, exact_duplicate_groups, near_duplicate_pairs
from src.data.schema import Record

SPLIT_NAMES = ("train", "validation", "test")


class UnionFind:
    """Minimal disjoint-set structure over record indices."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class SplitResult:
    train: list[Record]
    validation: list[Record]
    test: list[Record]
    groups: list[list[int]]
    report: dict[str, Any]

    def as_dict(self) -> dict[str, list[Record]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def build_groups(
    records: list[Record],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    group_similar: bool = True,
) -> list[list[int]]:
    """Return groups of record *positions* that must stay in the same split.

    Grouping links (a) exact duplicate instruction/response pairs, (b) exact
    duplicate instructions, and, when ``group_similar`` is set, (c) near-duplicate
    instructions above ``threshold``. Links are transitive.
    """
    position_of = {record.id: index for index, record in enumerate(records)}
    union_find = UnionFind(len(records))

    for group in exact_duplicate_groups(records, field="pair"):
        for other in group[1:]:
            union_find.union(position_of[group[0]], position_of[other])
    for group in exact_duplicate_groups(records, field="instruction"):
        for other in group[1:]:
            union_find.union(position_of[group[0]], position_of[other])

    if group_similar:
        for pair in near_duplicate_pairs(records, threshold=threshold, field="instruction"):
            union_find.union(position_of[pair.left_id], position_of[pair.right_id])

    clusters: dict[int, list[int]] = {}
    for index in range(len(records)):
        clusters.setdefault(union_find.find(index), []).append(index)
    # Deterministic order: largest first, then by smallest member index.
    return sorted(clusters.values(), key=lambda g: (-len(g), g[0]))


def assign_groups(
    groups: list[list[int]],
    ratios: dict[str, float],
    total: int,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Greedily assign whole groups to splits, largest group first.

    Each group goes to whichever split is currently furthest below its target
    count, so large groups cannot blow past a small split's quota.
    """
    targets = {name: ratios[name] * total for name in SPLIT_NAMES}
    assigned: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    counts = dict.fromkeys(SPLIT_NAMES, 0)

    rng = random.Random(seed)
    ordered = sorted(groups, key=lambda g: (-len(g), g[0]))
    # Shuffle within equal-size bands so the split is not an artefact of record order.
    bands: dict[int, list[list[int]]] = {}
    for group in ordered:
        bands.setdefault(len(group), []).append(group)
    ordered = []
    for size in sorted(bands, reverse=True):
        band = bands[size]
        rng.shuffle(band)
        ordered.extend(band)

    for group in ordered:
        deficits = {name: targets[name] - counts[name] for name in SPLIT_NAMES}
        # Tie-break by the fixed SPLIT_NAMES order to keep this deterministic.
        best = max(SPLIT_NAMES, key=lambda name: (deficits[name], -SPLIT_NAMES.index(name)))
        assigned[best].extend(group)
        counts[best] += len(group)
    return assigned


def split_records(
    records: list[Record],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    threshold: float = DEFAULT_THRESHOLD,
    group_similar: bool = True,
    seed: int = 42,
) -> SplitResult:
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total_ratio}")
    if not records:
        raise ValueError("cannot split an empty dataset")

    groups = build_groups(records, threshold=threshold, group_similar=group_similar)
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    assigned = assign_groups(groups, ratios, len(records), seed=seed)

    group_of_position = {pos: gid for gid, group in enumerate(groups) for pos in group}
    splits: dict[str, list[Record]] = {}
    for name in SPLIT_NAMES:
        splits[name] = [
            replace(records[pos], group_id=group_of_position[pos])
            for pos in sorted(assigned[name])
        ]

    sizes = [len(g) for g in groups]
    report = {
        "seed": seed,
        "requested_ratios": ratios,
        "near_duplicate_threshold": threshold,
        "group_similar_instructions": group_similar,
        "total_records": len(records),
        "total_groups": len(groups),
        "largest_group_size": max(sizes),
        "singleton_groups": sum(1 for s in sizes if s == 1),
        "multi_record_groups": sum(1 for s in sizes if s > 1),
        "split_sizes": {name: len(splits[name]) for name in SPLIT_NAMES},
        "actual_ratios": {
            name: round(len(splits[name]) / len(records), 4) for name in SPLIT_NAMES
        },
        "groups_per_split": {
            name: len({r.group_id for r in splits[name]}) for name in SPLIT_NAMES
        },
    }
    return SplitResult(splits["train"], splits["validation"], splits["test"], groups, report)
