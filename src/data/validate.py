"""Dataset-level validation and the human/machine readable quality report."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.schema import Record, ValidationIssue, load_records, normalize_text

# Rough heuristic used only for capacity planning (no tokenizer needed on CPU/CI).
CHARS_PER_TOKEN = 3.8


def _length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": ordered[p95_index],
    }


def dataset_hash(records: list[Record]) -> str:
    """Order-independent SHA-256 over the instruction/response content."""
    digest = hashlib.sha256()
    for line in sorted(f"{r.instruction}\x1f{r.response}" for r in records):
        digest.update(line.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def build_dataset_report(
    records: list[Record], issues: list[ValidationIssue], source: str
) -> dict[str, Any]:
    instruction_chars = [len(r.instruction) for r in records]
    response_chars = [len(r.response) for r in records]
    instruction_words = [len(r.instruction.split()) for r in records]
    response_words = [len(r.response.split()) for r in records]

    pair_keys = [(normalize_text(r.instruction), normalize_text(r.response)) for r in records]
    duplicate_records = len(pair_keys) - len(set(pair_keys))
    instruction_keys = [k[0] for k in pair_keys]
    duplicate_instructions = len(instruction_keys) - len(set(instruction_keys))

    total_chars = sum(instruction_chars) + sum(response_chars)
    return {
        "source": source,
        "total_records": len(records) + len(issues),
        "valid_records": len(records),
        "invalid_records": len(issues),
        "duplicate_records": duplicate_records,
        "duplicate_instructions": duplicate_instructions,
        "dataset_sha256": dataset_hash(records),
        "instruction_statistics": {
            "characters": _length_stats(instruction_chars),
            "words": _length_stats(instruction_words),
        },
        "response_statistics": {
            "characters": _length_stats(response_chars),
            "words": _length_stats(response_words),
        },
        "estimated_tokens": int(total_chars / CHARS_PER_TOKEN),
        "estimated_tokens_note": (
            f"Character-based approximation at ~{CHARS_PER_TOKEN} chars/token; "
            "no tokenizer is required so this runs in CPU-only CI."
        ),
        "issues_by_reason": dict(Counter(i.reason for i in issues)),
        "issues": [i.to_dict() for i in issues],
    }


def render_dataset_report(report: dict[str, Any]) -> str:
    """Human-readable version of the dataset report (no personal content echoed)."""
    lines = [
        "# Dataset Quality Report",
        "",
        f"Source: `{report['source']}`",
        f"SHA-256: `{report['dataset_sha256']}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total records | {report['total_records']} |",
        f"| Valid records | {report['valid_records']} |",
        f"| Invalid records | {report['invalid_records']} |",
        f"| Exact duplicate pairs | {report['duplicate_records']} |",
        f"| Repeated instructions | {report['duplicate_instructions']} |",
        f"| Estimated tokens | {report['estimated_tokens']} |",
        "",
        "## Length statistics",
        "",
        "| Field | Unit | Min | Max | Mean | Median | P95 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for field_name in ("instruction", "response"):
        stats = report[f"{field_name}_statistics"]
        for unit in ("characters", "words"):
            s = stats[unit]
            lines.append(
                f"| {field_name} | {unit} | {s['min']} | {s['max']} | "
                f"{s['mean']} | {s['median']} | {s['p95']} |"
            )
    if report["issues_by_reason"]:
        lines += ["", "## Rejected lines", "", "| Reason | Count |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(report["issues_by_reason"].items())]
    else:
        lines += ["", "All records passed schema validation."]
    return "\n".join(lines) + "\n"


def validate_dataset(path: str | Path, source: str | None = None) -> tuple[list[Record], dict[str, Any]]:
    records, issues = load_records(path)
    report = build_dataset_report(records, issues, source or str(path))
    return records, report
