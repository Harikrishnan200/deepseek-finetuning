#!/usr/bin/env python3
"""Publish a trained adapter to the Hugging Face Hub (reads HF_TOKEN from the env).

    export HF_TOKEN=hf_...
    python scripts/push_to_hub.py --hub-model-id YOUR_USERNAME/deepseek-personal-qlora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.publish import push_adapter  # noqa: E402
from src.training.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--adapter-path", default="artifacts/training/adapter")
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="owner/name. Omit the owner (or the whole flag) to use the account HF_TOKEN belongs to.",
    )
    parser.add_argument("--evaluation-dir", default="artifacts/evaluation")
    parser.add_argument("--public", action="store_true", help="create a public repo (default private)")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="refuse to publish unless the promotion gate verdict is PASS",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_file = Path("artifacts/training/run_metadata.json")
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}

    url = push_adapter(
        adapter_dir=args.adapter_path,
        model_id=args.hub_model_id,
        config=config,
        private=not args.public,
        metadata=metadata,
        evaluation_dir=args.evaluation_dir,
        require_pass=args.require_pass,
    )
    print(f"[hub] {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
