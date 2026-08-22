"""Deduplication, leakage-aware splitting, and split determinism."""

from __future__ import annotations

import pytest

from src.data.deduplicate import (
    build_duplicate_report,
    cross_similarity,
    exact_duplicate_groups,
    near_duplicate_pairs,
)
from src.data.leakage import analyse_split_leakage, exact_overlap
from src.data.schema import Record, normalize_text
from src.data.split import SPLIT_NAMES, assign_groups, build_groups, split_records

# ------------------------------------------------------------------ duplicates


def test_exact_duplicates_are_found(sample_records):
    groups = exact_duplicate_groups(sample_records, field="pair")
    assert [0, 2] in groups


def test_exact_duplicate_detection_ignores_case_and_punctuation():
    records = [
        Record("What is his name?", "Ada", 0),
        Record("what is his name", "ada", 1),
    ]
    assert exact_duplicate_groups(records, field="pair") == [[0, 1]]


def test_near_duplicates_found_above_threshold(sample_records):
    pairs = near_duplicate_pairs(sample_records, threshold=0.80, field="instruction")
    found = {(p.left_id, p.right_id) for p in pairs}
    assert (0, 1) in found, "'What is' vs 'What was' should be near duplicates"


def test_near_duplicate_threshold_is_monotonic(sample_records):
    loose = near_duplicate_pairs(sample_records, threshold=0.60)
    strict = near_duplicate_pairs(sample_records, threshold=0.95)
    assert len(loose) >= len(strict)


def test_duplicate_report_reports_without_deleting(sample_records):
    report = build_duplicate_report(sample_records, threshold=0.8)
    assert report["total_records"] == len(sample_records)
    assert report["exact_duplicate_redundant_records"] >= 1
    assert "no record is deleted automatically" in report["policy"]


def test_cross_similarity_corpus_makes_scores_stable():
    """Fitting IDF on a fixed corpus must give the same score for the same pair."""
    left = ["what grade did he get in physics"]
    right = ["did he study physics", "what is his name", "where does he live"]
    corpus = left + right + ["a totally unrelated sentence about trains"]
    full = cross_similarity(left, right, corpus=corpus)
    subset = cross_similarity(left, right[:1], corpus=corpus)
    assert full[0, 0] == pytest.approx(subset[0, 0], abs=1e-6)


# ---------------------------------------------------------------------- groups


def test_groups_partition_every_record(sample_records):
    groups = build_groups(sample_records, threshold=0.8)
    positions = sorted(p for g in groups for p in g)
    assert positions == list(range(len(sample_records)))


def test_grouping_keeps_duplicates_together(sample_records):
    groups = build_groups(sample_records, threshold=0.8)
    group_of = {p: i for i, g in enumerate(groups) for p in g}
    assert group_of[0] == group_of[2], "exact duplicates must share a group"
    assert group_of[0] == group_of[1], "near duplicates must share a group"


def test_grouping_can_be_disabled(sample_records):
    with_similar = build_groups(sample_records, threshold=0.8, group_similar=True)
    without = build_groups(sample_records, threshold=0.8, group_similar=False)
    assert len(without) >= len(with_similar)


def test_assign_groups_respects_targets():
    groups = [[i] for i in range(100)]
    assigned = assign_groups(groups, {"train": 0.7, "validation": 0.15, "test": 0.15}, 100, seed=42)
    assert len(assigned["train"]) == 70
    assert len(assigned["validation"]) == 15
    assert len(assigned["test"]) == 15


# ---------------------------------------------------------------------- splits


def test_split_ratios_are_approximately_correct(sample_records):
    result = split_records(sample_records, threshold=0.8, seed=42)
    total = len(sample_records)
    assert sum(len(s) for s in result.as_dict().values()) == total
    assert result.report["actual_ratios"]["train"] == pytest.approx(0.70, abs=0.15)


def test_split_is_deterministic(sample_records):
    a = split_records(sample_records, threshold=0.8, seed=42)
    b = split_records(sample_records, threshold=0.8, seed=42)
    for name in SPLIT_NAMES:
        assert [r.id for r in getattr(a, name)] == [r.id for r in getattr(b, name)]


def test_different_seeds_can_produce_different_splits(sample_records):
    a = split_records(sample_records, threshold=0.8, seed=42)
    b = split_records(sample_records, threshold=0.8, seed=1234)
    # Not guaranteed different, but the assignment must remain valid either way.
    for result in (a, b):
        assert sum(len(s) for s in result.as_dict().values()) == len(sample_records)


def test_no_record_appears_in_two_splits(sample_records):
    result = split_records(sample_records, threshold=0.8, seed=42)
    ids = [r.id for split in result.as_dict().values() for r in split]
    assert len(ids) == len(set(ids))


def test_no_exact_train_test_overlap(sample_records):
    result = split_records(sample_records, threshold=0.8, seed=42)
    assert exact_overlap(result.test, result.train) == []
    train_keys = {normalize_text(r.instruction) for r in result.train}
    test_keys = {normalize_text(r.instruction) for r in result.test}
    assert not (train_keys & test_keys)


def test_no_group_spans_two_splits(sample_records):
    result = split_records(sample_records, threshold=0.8, seed=42)
    seen: dict[int, str] = {}
    for name, split in result.as_dict().items():
        for record in split:
            assert seen.setdefault(record.group_id, name) == name


def test_leakage_analysis_is_clean_after_grouped_split(sample_records):
    result = split_records(sample_records, threshold=0.8, seed=42)
    report = analyse_split_leakage(result.train, result.validation, result.test, threshold=0.8)
    assert report["leakage_free"], report["splits"]
    assert report["max_overlap_rate"] == 0.0


def test_leakage_analysis_detects_a_planted_overlap():
    train = [Record("What is his full name?", "Ada Lovelace", 0, 0)]
    test = [Record("what is his full name", "Ada Lovelace", 1, 1)]
    report = analyse_split_leakage(train, [], test, threshold=0.9)
    assert not report["leakage_free"]
    assert report["splits"]["test"]["exact_overlaps"] == 1


def test_invalid_ratios_rejected(sample_records):
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_records(sample_records, train_ratio=0.8, validation_ratio=0.15, test_ratio=0.15)


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        split_records([])
