#!/usr/bin/env bash
# Build Klein i2i JSONL from ToolGen difficulty_label_shards (visual difficulty >= threshold).
# Wraps: scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py (multiprocessing + tqdm).
#
# Pipeline order (Python):
#   1) Emit all matching rows to train.jsonl (combined pool).
#   2) Normalize every condition image to NORMALIZE_SIZE x NORMALIZE_SIZE (white letterbox), write PNGs
#      under <dataset>/CONDITION_IMAGES_SUBDIR/, rewrite JSONL paths to those files; drop rows on failure.
#   3) Prune broken / oversize references (same as before) on the combined file — before split.
#   4) Train/test split: exactly SPLIT_TEST_COUNT rows (default 64) -> test.jsonl; remainder -> train.jsonl.
#
# Environment (optional):
#   FLOW_FACTORY_ROOT   Repo root (default: parent of this script directory); exported for subprocesses
#   TOOLGEN_ROOT        ToolGen checkout (default below)
#   SHARD_DIR           difficulty_label_shards directory (default: under ToolGen phase2)
#   OUTPUT_JSONL        Output path (default: dataset/klein_i2i_visual_ge_35_v3/train.jsonl)
#   MIN_VISUAL_DIFF     Default 3.5
#   WORKERS             0 = auto (min(32, CPUs)); 1 = single-process
#   MAX_ROWS            0 = write all matching rows; else cap emitted lines
#   MAX_SHARD_FILES     0 = all shard JSON files; else dev cap on files scanned
#
#   SPLIT_TEST_COUNT    Default 64. 0 = no test split, still run normalize+prune on train.jsonl only.
#   SPLIT_RANDOM        Default 1 (set 0 for first-N test lines, no shuffle).
#   SPLIT_SEED          Default 42 (with SPLIT_RANDOM=1).
#   NO_PRUNE            Set to 1 to skip the prune pass.
#
#   NO_NORMALIZE        Set to 1 to skip white-canvas resize (paths stay on original refs).
#   NORMALIZE_SIZE      Default 512
#   CONDITION_IMAGES_SUBDIR  Subfolder under dataset root for PNGs (default: condition_images_512)
#
# Usage:
#   bash scripts/build_klein_i2i_rl_dataset.sh
#   OUTPUT_JSONL=/data/klein_v3/train.jsonl WORKERS=16 bash scripts/build_klein_i2i_rl_dataset.sh
#   SPLIT_TEST_COUNT=0 bash scripts/build_klein_i2i_rl_dataset.sh   # build only, no split
# Pass-through Python flags (e.g. --no-progress):
#   bash scripts/build_klein_i2i_rl_dataset.sh --no-progress

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_FACTORY_ROOT="$(cd "${HERE}/.." && pwd)"
export FLOW_FACTORY_ROOT

TOOLGEN_ROOT="${TOOLGEN_ROOT:-/primus_xpfs_workspace_T04/haozhe/ToolGen}"
SHARD_DIR="${SHARD_DIR:-${TOOLGEN_ROOT}/phase2_prompt_generation/difficulty_label_shards}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${FLOW_FACTORY_ROOT}/dataset/klein_i2i_visual_ge_35_v3/train.jsonl}"
MIN_VISUAL_DIFF="${MIN_VISUAL_DIFF:-3.5}"
WORKERS="${WORKERS:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
MAX_SHARD_FILES="${MAX_SHARD_FILES:-0}"

SPLIT_TEST_COUNT="${SPLIT_TEST_COUNT:-64}"
SPLIT_RANDOM="${SPLIT_RANDOM:-1}"
SPLIT_SEED="${SPLIT_SEED:-42}"

NORMALIZE_SIZE="${NORMALIZE_SIZE:-512}"
CONDITION_IMAGES_SUBDIR="${CONDITION_IMAGES_SUBDIR:-condition_images_512}"

export PYTHONPATH="${FLOW_FACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

py_args=(
  "${FLOW_FACTORY_ROOT}/scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py"
  --toolgen-root "${TOOLGEN_ROOT}"
  --shard-dir "${SHARD_DIR}"
  --output "${OUTPUT_JSONL}"
  --min-visual-difficulty "${MIN_VISUAL_DIFF}"
  --workers "${WORKERS}"
  --test-count "${SPLIT_TEST_COUNT}"
  --split-seed "${SPLIT_SEED}"
  --normalize-size "${NORMALIZE_SIZE}"
  --normalize-subdir "${CONDITION_IMAGES_SUBDIR}"
)
if [[ "${SPLIT_RANDOM}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--split-random)
fi
if [[ "${NO_PRUNE:-0}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--no-prune)
fi
if [[ "${NO_NORMALIZE:-0}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--no-normalize-condition-images)
fi

if [[ "${MAX_ROWS}" != "0" ]]; then
  py_args+=(--max-rows "${MAX_ROWS}")
fi
if [[ "${MAX_SHARD_FILES}" != "0" ]]; then
  py_args+=(--max-shard-files "${MAX_SHARD_FILES}")
fi

exec python3 "${py_args[@]}" "$@"
