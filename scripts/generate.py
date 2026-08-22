#!/usr/bin/env python3
"""Generate answers from the fine-tuned model.

    python scripts/generate.py --adapter-path artifacts/training/adapter \
        --prompt "What is Harikrishnan's full name?"

    python scripts/generate.py --interactive
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
    parser.add_argument("--adapter-path", default="artifacts/training/adapter")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 = greedy/deterministic"
    )
    parser.add_argument("--base-only", action="store_true", help="skip the adapter")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        parser.error("provide --prompt or --interactive")

    config = load_config(args.config)
    from src.inference.generate import Generator, interactive

    generator = Generator.from_config(config, None if args.base_only else args.adapter_path)

    if args.interactive:
        interactive(generator, max_new_tokens=args.max_new_tokens)
        return 0

    result = generator.generate(
        args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.response)
        print(
            f"\n[{result.latency_seconds:.2f}s, {result.generated_tokens} tokens, "
            f"{result.tokens_per_second} tok/s]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
