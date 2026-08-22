#!/usr/bin/env python3
"""Validate the raw dataset schema and emit the quality report.

    python scripts/validate_dataset.py --input data/raw/personal_dataset.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import write_json  # noqa: E402
from src.data.validate import render_dataset_report, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/personal_dataset.jsonl")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero if any record fails validation"
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[skip] dataset not found: {args.input}")
        print("       This is expected in CI - the personal dataset is not committed.")
        return 0

    records, report = validate_dataset(args.input)
    write_json(Path(args.report_dir) / "dataset_report.json", report)
    Path(args.report_dir, "dataset_report.md").write_text(
        render_dataset_report(report), encoding="utf-8"
    )

    print(render_dataset_report(report))
    if args.strict and report["invalid_records"]:
        print(f"[fail] {report['invalid_records']} invalid records", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
