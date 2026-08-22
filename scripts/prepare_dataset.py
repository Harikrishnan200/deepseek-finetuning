#!/usr/bin/env python3
"""Run the full data pipeline: validate, deduplicate, split, leakage-check.

    python scripts/prepare_dataset.py --config configs/qlora.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import prepare  # noqa: E402
from src.training.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--input", default=None, help="override config dataset.raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument(
        "--fail-on-leakage", action="store_true", help="exit non-zero if any overlap is found"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = args.input or config.dataset["raw"]
    if not Path(raw_path).exists():
        print(f"[skip] dataset not found: {raw_path}")
        return 0

    split_cfg = config.split
    summary = prepare(
        raw_path,
        train_ratio=split_cfg["train_ratio"],
        validation_ratio=split_cfg["validation_ratio"],
        test_ratio=split_cfg["test_ratio"],
        threshold=split_cfg["near_duplicate_threshold"],
        group_similar=split_cfg["group_similar_instructions"],
        seed=config.seed,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
    )

    dataset = summary["dataset_report"]
    split = summary["split_report"]
    leakage = summary["leakage_report"]

    print(f"valid records        : {dataset['valid_records']} (invalid: {dataset['invalid_records']})")
    print(f"exact duplicate pairs: {summary['duplicate_report']['exact_duplicate_redundant_records']}")
    print(f"near-duplicate pairs : {summary['duplicate_report']['near_duplicate_instruction_pairs']}")
    print(f"groups               : {split['total_groups']} (largest {split['largest_group_size']})")
    print(f"split sizes          : {split['split_sizes']}")
    print(f"actual ratios        : {split['actual_ratios']}")
    print(f"max overlap rate     : {leakage['max_overlap_rate']}")
    print(f"leakage free         : {leakage['leakage_free']}")
    print(f"\nwrote splits to {args.processed_dir}/ and reports to {args.report_dir}/")

    if args.fail_on_leakage and not leakage["leakage_free"]:
        print("[fail] cross-split leakage detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
