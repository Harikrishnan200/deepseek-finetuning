#!/usr/bin/env python3
"""Side-by-side base vs fine-tuned answers for a handful of prompts.

Complements scripts/evaluate.py: that one produces the numbers, this one lets you
read the actual outputs and judge response quality yourself.

    python scripts/compare_models.py --prompts "What is his full name?" "Where did he study?"
    python scripts/compare_models.py --from-split data/processed/test.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import load_records  # noqa: E402
from src.evaluation.metrics import score_pair  # noqa: E402
from src.training.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--adapter-path", default="artifacts/training/adapter")
    parser.add_argument("--prompts", nargs="*", default=[])
    parser.add_argument("--from-split", default=None, help="read prompts from a JSONL split")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    references: list[str | None] = []
    if args.from_split:
        records, _ = load_records(args.from_split)
        prompts = [r.instruction for r in records[: args.limit]]
        references = [r.response for r in records[: args.limit]]
    else:
        prompts = args.prompts
        references = [None] * len(prompts)
    if not prompts:
        parser.error("provide --prompts or --from-split")

    config = load_config(args.config)
    from src.inference.generate import Generator

    outputs: dict[str, list[str]] = {}
    for label, adapter in (("base", None), ("fine_tuned", args.adapter_path)):
        print(f"[load] {label} ...", file=sys.stderr)
        generator = Generator.from_config(config, adapter)
        outputs[label] = [
            generator.generate(p, max_new_tokens=args.max_new_tokens).response for p in prompts
        ]
        del generator

    for index, prompt in enumerate(prompts):
        print(f"\n{'=' * 78}\nQ: {prompt}")
        if references[index] is not None:
            print(f"REFERENCE   : {references[index]}")
        for label in ("base", "fine_tuned"):
            print(f"{label:<12}: {outputs[label][index]}")
            if references[index] is not None:
                scores = score_pair(outputs[label][index], references[index])
                print(
                    f"{'':12}  (norm EM {scores['normalized_exact_match']:.0f}, "
                    f"F1 {scores['token_f1']:.3f})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
