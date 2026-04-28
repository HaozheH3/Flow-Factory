#!/usr/bin/env bash
# Re-run the same filter on an existing dataset directory. The Klein i2i builder
# (scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py) already applies
# this after train/test split by default: --prune-max-images 3, --prune-drop-pixels-percentile 98.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python3 scripts/prune_jsonl_broken_reference_images.py \
    --dataset-dir dataset/klein_i2i_visual_ge_35 \
    --max-images 3 \
    --drop-pixels-above-percentile 98