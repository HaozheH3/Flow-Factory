#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Move the first N lines (or a random N) from a JSONL train file into a test file; rewrite train with the rest.

Example (Klein i2i dataset default layout):
  python scripts/split_jsonl_train_test.py \\
    --input dataset/klein_i2i_visual_ge_35/train.jsonl \\
    --test-count 100
Writes ``dataset/klein_i2i_visual_ge_35/test.jsonl`` and replaces ``train.jsonl`` with remaining rows.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List


def _read_nonempty_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            lines.append(s)
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Source train JSONL path.")
    p.add_argument("--test-count", type=int, default=100, help="Number of rows to move to test (default 100).")
    p.add_argument(
        "--test-output",
        type=Path,
        default=None,
        help="Test JSONL path (default: same directory as --input, named test.jsonl).",
    )
    p.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Train JSONL path after split (default: overwrite --input).",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Shuffle rows before splitting (use with --seed for reproducibility).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed when --random is set.")
    args = p.parse_args()

    src = args.input.resolve()
    if not src.is_file():
        raise SystemExit(f"Input not found: {src}")

    n_test = int(args.test_count)
    if n_test <= 0:
        raise SystemExit("--test-count must be positive")

    lines = _read_nonempty_lines(src)
    if len(lines) < n_test:
        raise SystemExit(
            f"Not enough rows: need at least {n_test}, got {len(lines)} in {src}"
        )

    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(lines)

    test_lines = lines[:n_test]
    train_lines = lines[n_test:]

    out_dir = src.parent
    test_path = (args.test_output or (out_dir / "test.jsonl")).resolve()
    train_path = (args.train_output or src).resolve()

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(test_path.parent),
        prefix=".test_split_",
        suffix=".jsonl",
        delete=False,
    ) as tf_test:
        test_tmp = Path(tf_test.name)
        for row in test_lines:
            tf_test.write(row + "\n")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(train_path.parent),
        prefix=".train_split_",
        suffix=".jsonl",
        delete=False,
    ) as tf_train:
        train_tmp = Path(tf_train.name)
        for row in train_lines:
            tf_train.write(row + "\n")

    shutil.move(str(test_tmp), str(test_path))
    shutil.move(str(train_tmp), str(train_path))

    print(
        f"Split {src.name}: wrote {len(test_lines)} rows -> {test_path}, "
        f"{len(train_lines)} rows -> {train_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
