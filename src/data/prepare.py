"""Pipeline orchestration: validate -> deduplicate -> split -> leakage-check -> write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.deduplicate import DEFAULT_THRESHOLD, build_duplicate_report
from src.data.leakage import analyse_split_leakage
from src.data.schema import Record, write_jsonl
from src.data.split import split_records
from src.data.validate import render_dataset_report, validate_dataset

REPORT_DIR = Path("data/reports")
PROCESSED_DIR = Path("data/processed")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_prompt(record: Record, template: str, *, for_inference: bool = False) -> str:
    """Render a record with the configured prompt template.

    With ``for_inference`` the response slot is left empty so the model completes it.
    """
    return template.format(
        instruction=record.instruction, response="" if for_inference else record.response
    )


def prepare(
    raw_path: str | Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    threshold: float = DEFAULT_THRESHOLD,
    group_similar: bool = True,
    seed: int = 42,
    processed_dir: str | Path = PROCESSED_DIR,
    report_dir: str | Path = REPORT_DIR,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the full data pipeline and return a summary of what happened."""
    processed_dir = Path(processed_dir)
    report_dir = Path(report_dir)

    records, dataset_report = validate_dataset(raw_path)
    if not records:
        raise SystemExit(f"No valid records found in {raw_path}")

    duplicate_report = build_duplicate_report(records, threshold=threshold)
    result = split_records(
        records,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        threshold=threshold,
        group_similar=group_similar,
        seed=seed,
    )
    leakage_report = analyse_split_leakage(
        result.train, result.validation, result.test, threshold=threshold
    )

    if write_outputs:
        for name, split in result.as_dict().items():
            write_jsonl(processed_dir / f"{name}.jsonl", split)
        write_json(report_dir / "dataset_report.json", dataset_report)
        write_json(report_dir / "duplicate_report.json", duplicate_report)
        write_json(
            report_dir / "leakage_report.json",
            {"split": result.report, "leakage": leakage_report},
        )
        (report_dir / "dataset_report.md").write_text(
            render_dataset_report(dataset_report), encoding="utf-8"
        )

    return {
        "dataset_report": dataset_report,
        "duplicate_report": duplicate_report,
        "split_report": result.report,
        "leakage_report": leakage_report,
        "splits": result.as_dict(),
    }
