#!/usr/bin/env bash
# Build Klein mixed T2I+I2I JSONL from ToolGen labeled JSONL (visual difficulty >= threshold) or from shards.
# Default source: phase2_prompt_generation/AA_synth_all_prompts_metadata_eval.difficulty_labeled.jsonl
#
# Pipeline order (Python):
#   1) Emit matching rows to train.jsonl (combined pool). Generator prompt = refined; user_prompt = original.
#   2) Normalize every condition image (I2I rows only) to NORMALIZE_SIZE² white letterbox; T2I rows unchanged.
#   3) Prune broken references (--allow-empty-images when mixing). Before split.
#   4) Train/test split: SPLIT_TEST_COUNT rows -> test.jsonl.
#
# Environment (optional):
#   FLOW_FACTORY_ROOT   Repo root (default: parent of this script directory)
#   TOOLGEN_ROOT        ToolGen checkout
#   LABELED_JSONL      Override path to difficulty_labeled.jsonl (default: under ToolGen phase2)
#   USE_SHARDS          Set to 1 to use SHARD_DIR instead of LABELED_JSONL
#   SHARD_DIR           difficulty_label_shards (when USE_SHARDS=1)
#   OUTPUT_JSONL        (default: dataset/klein_i2i_mix_visual_ge_35_v4/train.jsonl)
#   MIN_VISUAL_DIFF     Default 3.5
#   WORKERS            0 = auto
#   MAX_ROWS / MAX_SHARD_FILES / MAX_JSONL_LINES  dev caps (0 = all)
#   NO_MIX_T2I_I2I     Set to 1 for I2I-only (drop rows with no reference images)
#   SPLIT_TEST_COUNT / SPLIT_RANDOM / SPLIT_SEED / NO_PRUNE / NO_NORMALIZE / NORMALIZE_SIZE / CONDITION_IMAGES_SUBDIR
#
# Usage:
#   bash scripts/build_klein_i2i_rl_dataset.sh
#   OUTPUT_JSONL=/data/klein_v4/train.jsonl WORKERS=16 bash scripts/build_klein_i2i_rl_dataset.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_FACTORY_ROOT="$(cd "${HERE}/.." && pwd)"
export FLOW_FACTORY_ROOT

TOOLGEN_ROOT="${TOOLGEN_ROOT:-/primus_xpfs_workspace_T04/haozhe/ToolGen}"
LABELED_JSONL="${LABELED_JSONL:-${TOOLGEN_ROOT}/phase2_prompt_generation/AA_synth_all_prompts_metadata_eval.difficulty_labeled.jsonl}"
SHARD_DIR="${SHARD_DIR:-${TOOLGEN_ROOT}/phase2_prompt_generation/difficulty_label_shards}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${FLOW_FACTORY_ROOT}/dataset/klein_i2i_mix_visual_ge_35_v4/train.jsonl}"
MIN_VISUAL_DIFF="${MIN_VISUAL_DIFF:-3.5}"
WORKERS="${WORKERS:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
MAX_SHARD_FILES="${MAX_SHARD_FILES:-0}"
MAX_JSONL_LINES="${MAX_JSONL_LINES:-0}"
USE_SHARDS="${USE_SHARDS:-0}"

SPLIT_TEST_COUNT="${SPLIT_TEST_COUNT:-64}"
SPLIT_RANDOM="${SPLIT_RANDOM:-1}"
SPLIT_SEED="${SPLIT_SEED:-42}"

NORMALIZE_SIZE="${NORMALIZE_SIZE:-512}"
CONDITION_IMAGES_SUBDIR="${CONDITION_IMAGES_SUBDIR:-condition_images_512}"

export PYTHONPATH="${FLOW_FACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

py_args=(
  "${FLOW_FACTORY_ROOT}/scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py"
  --toolgen-root "${TOOLGEN_ROOT}"
  --output "${OUTPUT_JSONL}"
  --min-visual-difficulty "${MIN_VISUAL_DIFF}"
  --workers "${WORKERS}"
  --test-count "${SPLIT_TEST_COUNT}"
  --split-seed "${SPLIT_SEED}"
  --normalize-size "${NORMALIZE_SIZE}"
  --normalize-subdir "${CONDITION_IMAGES_SUBDIR}"
)

if [[ "${USE_SHARDS}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--shard-dir "${SHARD_DIR}")
else
  py_args+=(--labeled-jsonl "${LABELED_JSONL}")
fi

if [[ "${SPLIT_RANDOM}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--split-random)
fi
if [[ "${NO_PRUNE:-0}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--no-prune)
fi
if [[ "${NO_NORMALIZE:-0}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--no-normalize-condition-images)
fi
if [[ "${NO_MIX_T2I_I2I:-0}" =~ ^(1|true|yes)$ ]]; then
  py_args+=(--no-mix-t2i-i2i)
fi

if [[ "${MAX_ROWS}" != "0" ]]; then
  py_args+=(--max-rows "${MAX_ROWS}")
fi
if [[ "${MAX_SHARD_FILES}" != "0" ]]; then
  py_args+=(--max-shard-files "${MAX_SHARD_FILES}")
fi
if [[ "${MAX_JSONL_LINES}" != "0" ]]; then
  py_args+=(--max-jsonl-lines "${MAX_JSONL_LINES}")
fi

exec python3 "${py_args[@]}" "$@"
