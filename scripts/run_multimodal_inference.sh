#!/usr/bin/env bash
# Unified launcher for Flow-Factory companion inference scripts.
#
#   ./scripts/run_multimodal_inference.sh flux2-klein
#   ./scripts/run_multimodal_inference.sh bagel
#
# Flux (env-driven, same as run_flux2_klein_inference.sh):
#   MODEL_PATH=... STEPS=4 CPU_OFFLOAD=1 ./scripts/run_multimodal_inference.sh flux2-klein
#
# BAGEL (requires cloned https://github.com/bytedance-seed/BAGEL + its deps):
#   BAGEL_REPO=~/BAGEL BAGEL_MODEL_PATH=.../bagel-7b-mot ./scripts/run_multimodal_inference.sh bagel
#
# Extra CLI after the backend name is forwarded to Python, e.g.:
#   ./scripts/run_multimodal_inference.sh bagel --prompt "a cat" --mode 2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND="${1:-flux2-klein}"
shift || true

case "$BACKEND" in
  flux2-klein)
    export MODEL_PATH="${MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/black-forest-labs/FLUX.2-klein-base-9B}"
    ARGS=(
      inference/generate_multimodal.py
      flux2-klein
      --model-path "$MODEL_PATH"
      --prompt "${PROMPT:-A red panda sleeping on a mossy log, soft light, 85mm}"
      --output "${OUTPUT:-flux2_klein_out.png}"
      --height "${HEIGHT:-1024}"
      --width "${WIDTH:-1024}"
      --guidance-scale "${GUIDANCE:-4.0}"
      --seed "${SEED:-0}"
    )
    if [[ -n "${STEPS:-}" ]]; then
      ARGS+=(--num-inference-steps "$STEPS")
    fi
    if [[ -n "${CPU_OFFLOAD:-}" ]]; then
      ARGS+=(--cpu-offload)
    fi
    if [[ -n "${COMPILE:-}" ]]; then
      ARGS+=(--compile)
    fi
    if [[ -n "${COND_IMAGE:-}" ]]; then
      ARGS+=(--image "$COND_IMAGE")
    fi
    ARGS+=("$@")
    python "${ARGS[@]}"
    ;;
  bagel|bagel-7b|bagel-7b-mot)
    export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/bagel-7b-mot}"
    ARGS=(
      inference/generate_multimodal.py
      bagel
      --model-path "${BAGEL_MODEL_PATH}"
      --prompt "${PROMPT:-A red panda sleeping on a mossy log, soft light, 85mm}"
      --output "${OUTPUT:-bagel_out.png}"
      --seed "${SEED:-0}"
      --mode "${BAGEL_MODE:-1}"
    )
    if [[ -n "${BAGEL_REPO:-}" ]]; then
      ARGS+=(--bagel-repo "$BAGEL_REPO")
    fi
    if [[ -n "${BAGEL_CFG_TEXT:-}" ]]; then
      ARGS+=(--cfg-text-scale "$BAGEL_CFG_TEXT")
    fi
    if [[ -n "${BAGEL_STEPS:-}" ]]; then
      ARGS+=(--num-timesteps "$BAGEL_STEPS")
    fi
    ARGS+=("$@")
    python "${ARGS[@]}"
    ;;
  -h|--help|help)
    head -35 inference/generate_multimodal.py
    echo ""
    echo "Shell wrappers:"
    echo "  $0 flux2-klein   # env: MODEL_PATH, PROMPT, OUTPUT, STEPS, ..."
    echo "  $0 bagel         # env: BAGEL_REPO (required), BAGEL_MODEL_PATH, BAGEL_MODE, ..."
    ;;
  *)
    echo "Unknown backend: $BACKEND" >&2
    echo "Use: flux2-klein | bagel   (or: $0 --help)" >&2
    exit 1
    ;;
esac
