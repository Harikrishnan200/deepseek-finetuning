#!/usr/bin/env python3
"""Run QLoRA supervised fine-tuning. Requires a CUDA GPU (use Kaggle's free tier).

    python scripts/train_qlora.py --config configs/qlora.yaml
    python scripts/train_qlora.py --config configs/qlora.yaml \
        --push-to-hub --hub-model-id YOUR_USERNAME/deepseek-personal-qlora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--output-dir", default=None, help="override training.output_dir")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--private", action="store_true", help="create the Hub repo as private")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the config and exit without training"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[config] {args.config} -> run '{config.run_name}'")
    print(f"[config] model={config.model_name} seed={config.seed}")
    print(f"[config] effective batch size = {config.effective_batch_size}")
    if args.dry_run:
        print("[dry-run] config is valid; exiting without training.")
        return 0

    import torch

    if not torch.cuda.is_available():
        print(
            "[error] No CUDA GPU detected. QLoRA needs bitsandbytes 4-bit kernels, "
            "which are GPU-only.\n"
            "        Run this on Kaggle (notebooks/kaggle_train.ipynb) or another CUDA machine.\n"
            "        Use --dry-run to validate the config on CPU.",
            file=sys.stderr,
        )
        return 2
    print(f"[gpu] {torch.cuda.get_device_name(0)}")

    from src.training.trainer import train

    output_dir = Path(args.output_dir or config.training["output_dir"])
    result = train(config, output_dir=output_dir)
    print(f"\n[done] adapter saved to {result['adapter_path']}")

    # Loss curve straight after training, so it exists even if evaluation fails.
    from src.evaluation.plots import plot_loss_curves

    curve = plot_loss_curves(result["history"], output_dir / "loss_curve.png")
    if curve:
        print(f"[plot] {curve}")
    print(f"[history] {output_dir / 'training_history.json'}")
    print(f"[metadata] {output_dir / 'run_metadata.json'}")

    if args.push_to_hub:
        from src.publish import push_adapter

        model_id = args.hub_model_id or (config.hub or {}).get("model_id")
        if not model_id:
            print(
                "[error] --push-to-hub needs --hub-model-id (or hub.model_id in the config)",
                file=sys.stderr,
            )
            return 1
        url = push_adapter(
            adapter_dir=result["adapter_path"],
            model_id=model_id,
            config=config,
            private=args.private or bool((config.hub or {}).get("private", True)),
            metadata=result["metadata"],
        )
        print(f"[hub] pushed to {url}")

    print("\nNext: python scripts/evaluate.py --config", args.config)
    print(json.dumps(result["parameter_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
