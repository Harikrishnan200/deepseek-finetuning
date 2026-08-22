"""Schema validation, normalization, and JSONL IO."""

from __future__ import annotations

import json

import pytest

from src.data.schema import Record, load_records, normalize_text, validate_record, write_jsonl


def test_valid_record_parses():
    line = json.dumps({"instruction": "  Who is Ada?  ", "response": " Ada Lovelace. "})
    record, issue = validate_record(line, 1)
    assert issue is None
    assert record.instruction == "Who is Ada?"
    assert record.response == "Ada Lovelace."


@pytest.mark.parametrize(
    "line,reason",
    [
        ("{not valid json", "invalid_json"),
        ('["a", "b"]', "not_an_object"),
        ('{"instruction": "hi"}', "missing_field"),
        ('{"response": "hi"}', "missing_field"),
        ('{"instruction": 42, "response": "hi"}', "wrong_type"),
        ('{"instruction": "hi", "response": null}', "wrong_type"),
        ('{"instruction": "   ", "response": "hi"}', "empty_field"),
        ('{"instruction": "hi", "response": ""}', "empty_field"),
        ('{"instruction": "hi", "response": "yo", "secret": "x"}', "unknown_fields"),
    ],
)
def test_malformed_records_are_rejected(line, reason):
    record, issue = validate_record(line, 7)
    assert record is None
    assert issue.reason == reason
    assert issue.line_number == 7


def test_issue_dict_does_not_leak_raw_content():
    """Rejected lines may contain personal data - it must not reach the report."""
    line = '{"instruction": "SECRET PERSONAL DETAIL", "response": ""}'
    _, issue = validate_record(line, 1)
    assert "SECRET PERSONAL DETAIL" not in json.dumps(issue.to_dict())


def test_optional_category_field_allowed():
    record, issue = validate_record(
        '{"instruction": "2+2?", "response": "4", "category": "math"}', 1
    )
    assert issue is None
    assert record.category == "math"


def test_normalize_text():
    assert normalize_text("What is HIS full name?") == "what is his full name"
    assert normalize_text("  a,  b.  c! ") == "a b c"
    assert normalize_text("Café") == "cafe"


def test_load_records_skips_blank_lines_and_assigns_ids(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"instruction": "a", "response": "1"})
        + "\n\n   \n"
        + json.dumps({"instruction": "b", "response": "2"})
        + "\n",
        encoding="utf-8",
    )
    records, issues = load_records(path)
    assert not issues
    assert [r.id for r in records] == [0, 1]


def test_load_records_collects_issues_without_stopping(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"instruction": "a", "response": "1"}\nBROKEN\n{"instruction": "b", "response": "2"}\n',
        encoding="utf-8",
    )
    records, issues = load_records(path)
    assert len(records) == 2
    assert len(issues) == 1
    assert issues[0].line_number == 2


def test_write_then_load_roundtrip(tmp_path):
    records = [Record("q1", "a1", 0, 3), Record("q2", "a2", 1, 4)]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, records)
    loaded, issues = load_records(path)
    assert not issues
    assert [(r.instruction, r.response, r.id, r.group_id) for r in loaded] == [
        ("q1", "a1", 0, 3),
        ("q2", "a2", 1, 4),
    ]


def test_real_dataset_is_valid_if_present():
    """The committed splits (when present) must always pass validation."""
    from pathlib import Path

    path = Path("data/processed/test.jsonl")
    if not path.exists():
        import pytest

        pytest.skip("processed splits not present")
    records, issues = load_records(path)
    assert not issues
    assert records
