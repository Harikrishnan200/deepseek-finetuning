#!/usr/bin/env python3
"""Alias for scripts/prepare_dataset.py - splitting is one stage of the same pipeline.

    python scripts/split_dataset.py --config configs/qlora.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_dataset import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
