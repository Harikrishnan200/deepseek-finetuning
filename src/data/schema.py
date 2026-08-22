"""Dataset schema, normalization, and JSONL IO.

The on-disk format is JSONL with one object per line:

    {"instruction": "...", "response": "..."}

Processed splits carry two extra bookkeeping fields, ``id`` and ``group_id``,
which are added by the splitter and are ignored by the schema validator.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("instruction", "response")
OPTIONAL_FIELDS = ("id", "group_id", "category")

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Record:
    """A single validated instruction/response pair."""

    instruction: str
    response: str
    id: int = -1
    group_id: int = -1
    # Free-form tag used by the general-capability benchmark (math/reasoning/...).
    category: str = ""

    def to_dict(self, include_meta: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {"instruction": self.instruction, "response": self.response}
        if include_meta:
            d["id"] = self.id
            d["group_id"] = self.group_id
        if self.category:
            d["category"] = self.category
        return d


@dataclass
class ValidationIssue:
    """One reason a raw line was rejected."""

    line_number: int
    reason: str
    detail: str = ""
    raw: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        # `raw` is deliberately excluded: it may contain personal information.
        return {"line_number": self.line_number, "reason": self.reason, "detail": self.detail}


def normalize_text(text: str, *, strip_punctuation: bool = True) -> str:
    """Lowercase, strip accents/punctuation, and collapse whitespace.

    Used for duplicate detection, leakage detection, and normalized exact match,
    so that "What is his full name?" and "what is his full name" compare equal.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, raw_line)`` for every non-blank line."""
    with open(path, encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            if raw.strip():
                yield line_number, raw


def validate_record(raw: str, line_number: int) -> tuple[Record | None, ValidationIssue | None]:
    """Parse and validate one JSONL line.

    Returns ``(record, None)`` on success and ``(None, issue)`` on failure.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, ValidationIssue(line_number, "invalid_json", str(exc), raw)

    if not isinstance(obj, dict):
        return None, ValidationIssue(line_number, "not_an_object", type(obj).__name__, raw)

    for key in REQUIRED_FIELDS:
        if key not in obj:
            return None, ValidationIssue(line_number, "missing_field", key, raw)
        if not isinstance(obj[key], str):
            return None, ValidationIssue(
                line_number, "wrong_type", f"{key} is {type(obj[key]).__name__}, expected str", raw
            )
        if not obj[key].strip():
            return None, ValidationIssue(line_number, "empty_field", key, raw)

    unknown = sorted(set(obj) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        return None, ValidationIssue(line_number, "unknown_fields", ", ".join(unknown), raw)

    record = Record(
        instruction=obj["instruction"].strip(),
        response=obj["response"].strip(),
        id=int(obj.get("id", -1)),
        group_id=int(obj.get("group_id", -1)),
        category=str(obj.get("category", "")),
    )
    return record, None


def load_records(path: str | Path) -> tuple[list[Record], list[ValidationIssue]]:
    """Validate a whole JSONL file, returning valid records and rejected lines.

    Record ids are assigned by position among the *valid* records unless the file
    already carries explicit ids (processed splits do).
    """
    records: list[Record] = []
    issues: list[ValidationIssue] = []
    for line_number, raw in iter_jsonl(path):
        record, issue = validate_record(raw, line_number)
        if issue is not None:
            issues.append(issue)
            continue
        assert record is not None
        if record.id < 0:
            record = replace(record, id=len(records))
        records.append(record)
    return records, issues


def write_jsonl(path: str | Path, records: list[Record], *, include_meta: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(include_meta=include_meta), ensure_ascii=False))
            fh.write("\n")
