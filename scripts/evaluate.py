#!/usr/bin/env python3
"""Final evaluation: base vs fine-tuned on the untouched test set, plus the gate.

    python scripts/evaluate.py --config configs/qlora.yaml \
        --adapter-path artifacts/training/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.config import load_config, load_evaluation_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--eval-config", default="configs/evaluation.yaml")
    parser.add_argument("--adapter-path", default="artifacts/training/adapter")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--training-dir", default="artifacts/training")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="write model answers to disk (contains personal information; gitignored)",
    )
    parser.add_argument("--fail-on-gate", action="store_true", help="exit 1 unless the gate PASSes")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_config = load_evaluation_config(args.eval_config)

    if not Path(args.adapter_path).exists():
        print(f"[error] adapter not found: {args.adapter_path}", file=sys.stderr)
        print("        Train first: python scripts/train_qlora.py --config " + args.config, file=sys.stderr)
        return 2

    import torch

    if not torch.cuda.is_available():
        print(
            "[warn] no GPU detected - evaluation will be extremely slow on CPU "
            "and 4-bit quantization is unavailable.",
            file=sys.stderr,
        )

    from src.evaluation.evaluate import run_full_evaluation
    from src.evaluation.plots import generate_all_plots

    training_dir = Path(args.training_dir)
    results = run_full_evaluation(
        config,
        eval_config,
        args.adapter_path,
        output_dir=args.output_dir,
        training_history_path=training_dir / "training_history.json",
        run_metadata_path=training_dir / "run_metadata.json",
        save_predictions=args.save_predictions,
    )

    history_file = training_dir / "training_history.json"
    history = json.loads(history_file.read_text())["history"] if history_file.exists() else []
    plots = generate_all_plots(
        results, history, plots_dir=Path(args.output_dir) / "plots", training_dir=training_dir
    )
    for name, path in plots.items():
        print(f"[plot] {name}: {path}")

    print(f"\n[report] {Path(args.output_dir) / 'final_report.md'}")
    print(f"[verdict] {results['gate']['verdict']} - {results['gate']['explanation']}")

    if args.fail_on_gate and results["gate"]["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
