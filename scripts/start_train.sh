#!/usr/bin/env bash
# Local launcher for NFT + Rational Rewards T2I on the official Flow-Factory tree.
#
# Prerequisite: start the judge server (OpenAI-compatible) before training, e.g.
#   export CUDA_VISIBLE_DEVICES=0,1
#   export MODEL_PATH="/primus_xpfs_workspace_T04/ghl/models/TIGER-Lab/RationalRewards-8B-T2I"
#   ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
#
# Default config uses **local** base weights under LOCAL_MODELS_ROOT (see
# examples/nft/lora/qwen_image_rational_rewards_t2i_local_models.yaml). Set
# api_base_url / vlm_model in that YAML to match your vLLM `--served-model-name`.
#
# Usage:
#   ./scripts/start_train_rational_rewards_t2i.sh
#   CONFIG=examples/nft/lora/flux1_rational_rewards_t2i.yaml ./scripts/start_train_rational_rewards_t2i.sh
#   TRAIN_LOG=/path/to/run.log ./scripts/start_train_rational_rewards_t2i.sh
#   TRAIN_LOG=/path/to/run.log TRAIN_LOG_TEE=0 ./scripts/start_train_rational_rewards_t2i.sh   # file only, no tee
#
# Optional env:
#   NUM_PROCESSES       If set, passed as `--num_processes` to the launcher (overrides YAML). If unset, YAML + visible GPUs apply.
#   FLOWFACTORY_PYTHON  Python interpreter (default: same as this script's bash → `python3` / PATH)
#   CONFIG              Training YAML under repo root (default: qwen_image_rational_rewards_t2i_local_models.yaml)
#   LOCAL_MODELS_ROOT   Directory with mirrored HF-style trees (default: /primus_xpfs_workspace_T04/ghl/models)
#   FLOW_CACHE_ROOT     Repo-local tmp/torch/logs (default: <repo>/.runtime_cache)
#   TRAIN_LOG           Destination file for stdout+stderr (default: FLOW_CACHE_ROOT/logs/train_rational_rewards_t2i_<timestamp>.log)
#   TRAIN_LOG_TEE       If 1 (default), copy logs to the file and the terminal; if 0, file only
#   SKIP_TRAIN_LOG      If 1, do not redirect (terminal only)
#
# SwanLab (when YAML log.logging_backend is swanlab): set secrets in the shell or a gitignored file.
#   SWANLAB_API_KEY     optional override (default: key embedded below for one-command runs)
#   SWANLAB_MODE        cloud | local | offline (default: offline)
#   SWANLAB_SAVE_DIR    local run artifacts (default: <repo>/swanlab_logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Edit these for your cluster / home layout
# ---------------------------------------------------------------------------
# If your training env is not default `python3`, set e.g.:
#   export FLOWFACTORY_PYTHON="/primus_xpfs_workspace_T04/haozhe/flow_env/bin/python"
FLOWFACTORY_PYTHON="${FLOWFACTORY_PYTHON:-/primus_xpfs_workspace_T04/haozhe/flow_env/bin/python3}"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/primus_xpfs_workspace_T04/ghl/models}"

# Default config: local Qwen-Image weights + rational_rewards_t2i
CONFIG="${CONFIG:-/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/nft/lora/flux2_klein_judge_frontier_new_dist.yaml}"
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/nft/lora/qwen1edit_new.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/nft/lora/flux2_klein_judge_frontier_new.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo_warmstart_lora.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/grpo/lora/flux2_klein_judge_frontier.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/full/flux2_klein_toolgen_judge_dpo.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/grpo/lora/flux2_klein_toolgen_judge_grpo.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo_preference_sim_vllm.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo_preference_sim_frontier_warmstart.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_20k_dpo_sft_candidate.yaml
CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/grpo/lora/flux2_klein_judge_frontier_sft_candidate.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/dpo/lora/flux2_klein_toolgen_judge_dpo.yaml
# CONFIG=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/examples/grpo/full/flux2_klein_judge_frontier.yaml
# Ephemeral / bulky caches: keep under repo .runtime_cache; Hugging Face hub reads use LOCAL_MODELS_ROOT
FLOW_CACHE_ROOT="${FLOW_CACHE_ROOT:-${REPO_ROOT}/.runtime_cache}"
mkdir -p "${FLOW_CACHE_ROOT}/"{tmp,torch,xdg,logs}

export HF_HOME="${HF_HOME:-${LOCAL_MODELS_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${FLOW_CACHE_ROOT}/xdg}"
export TORCH_HOME="${TORCH_HOME:-${FLOW_CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${FLOW_CACHE_ROOT}/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TMPDIR}"

if [[ "${SKIP_TRAIN_LOG:-0}" != "1" ]]; then
  TRAIN_LOG="${TRAIN_LOG:-${FLOW_CACHE_ROOT}/logs/train_ff_$(date +%Y%m%d_%H%M%S).log}"
  mkdir -p "$(dirname "${TRAIN_LOG}")"
  # Create a stable symlink so the user can always find the current log
  ln -sfn "${TRAIN_LOG}" "${FLOW_CACHE_ROOT}/logs/latest.log"
  if [[ "${TRAIN_LOG_TEE:-1}" == "1" ]]; then
    echo "[flow-factory] logging stdout+stderr to ${TRAIN_LOG} (and terminal)"
    echo "[flow-factory] latest log symlink: ${FLOW_CACHE_ROOT}/logs/latest.log"
    exec > >(tee -a "${TRAIN_LOG}") 2>&1
  else
    echo "[flow-factory] logging stdout+stderr to ${TRAIN_LOG} (file only)" >&2
    echo "[flow-factory] latest log symlink: ${FLOW_CACHE_ROOT}/logs/latest.log" >&2
    exec >>"${TRAIN_LOG}" 2>&1
  fi
fi

# Shorter temp path helps some multiprocessing AF_UNIX limits (optional)
if [[ -d /tmp ]]; then
  mkdir -p /tmp/fftmp
  export TMPDIR=/tmp/fftmp
  export TMP=/tmp/fftmp
  export TEMP=/tmp/fftmp
fi

# Uncomment if your network requires a proxy for Hugging Face downloads
# export http_proxy=http://HOST:PORT
# export https_proxy=http://HOST:PORT

# SwanLab — defaults match private flowfactory_adapted/start_train.sh (override via env if needed).
# Security: remove or rotate the key before pushing to a shared or public remote.
export SWANLAB_SAVE_DIR="${SWANLAB_SAVE_DIR:-${REPO_ROOT}/swanlab_logs}"
mkdir -p "${SWANLAB_SAVE_DIR}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-kH9nbmAB0DrKA1aplxbjO}"

# Ensure editable install is importable when not using the same interpreter that has console_scripts
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

hash -r
echo "[flow-factory] REPO_ROOT=${REPO_ROOT}"
echo "[flow-factory] LOCAL_MODELS_ROOT=${LOCAL_MODELS_ROOT}"
echo "[flow-factory] FLOW_CACHE_ROOT=${FLOW_CACHE_ROOT}"
echo "[flow-factory] python: $(command -v "${FLOWFACTORY_PYTHON}")"
"${FLOWFACTORY_PYTHON}" -c "import sys; print('[flow-factory] version', sys.version.split()[0])"

if [[ ! -f "${CONFIG}" ]]; then
  echo "error: config not found: ${CONFIG} (cwd=$(pwd))" >&2
  exit 1
fi

# if [[ "${CONFIG}" == *"_local_models.yaml" ]] || [[ "${CONFIG}" == *"local_models"* ]]; then
#   _qwen_local="${LOCAL_MODELS_ROOT}/Qwen/Qwen-Image-2512"
#   if [[ ! -d "${_qwen_local}" ]]; then
#     echo "error: expected local Qwen-Image tree missing: ${_qwen_local}" >&2
#     echo "hint: set LOCAL_MODELS_ROOT to the parent of Qwen/Qwen-Image-2512" >&2
#     exit 1
#   fi
# fi

echo "[flow-factory] training with config: ${CONFIG}"

# ---------------------------------------------------------------------------
# Multi-GPU: flow_factory.cli must spawn `accelerate launch`. It uses a
# single-process "Direct launch" when RANK is set in the environment—even
# RANK=0 from Slurm or a parent shell—causing World Size 1 and one busy GPU.
# Clear stale single-rank markers so the YAML `num_processes` (or --num_processes
# below) actually runs multi-process training.
# ---------------------------------------------------------------------------
unset RANK WORLD_SIZE LOCAL_RANK OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE 2>/dev/null || true

_EXTRA_CLI=()
if [[ -n "${NUM_PROCESSES:-}" ]]; then
  _EXTRA_CLI+=(--num_processes "${NUM_PROCESSES}")
  echo "[flow-factory] --num_processes ${NUM_PROCESSES} (CLI overrides YAML)"
fi

exec "${FLOWFACTORY_PYTHON}" -m flow_factory.cli "${CONFIG}" "${_EXTRA_CLI[@]}" "$@"
