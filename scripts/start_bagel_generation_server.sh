#!/usr/bin/env bash
# Start ToolGen-compatible BAGEL-7B-MoT HTTP generation server (POST /, same JSON as generation_apis).
#
# Requires:
#   - Clone https://github.com/bytedance-seed/BAGEL and install its dependencies.
#   - Weights under BAGEL_MODEL_PATH (ema.safetensors, ae.safetensors, configs, tokenizer).
#
# Example:
#   export BAGEL_REPO=/path/to/BAGEL
#   export BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT-snapshot
#   ./scripts/start_bagel_generation_server.sh
#
# Multi-GPU + round-robin gateway (like Klein): scripts/start_bagel_8gpu_2workers_each.sh
#
# Optional:
#   BAGEL_SERVER_HOST  BAGEL_SERVER_PORT (default 8875)
#   BAGEL_PUBLIC_BASE_URL  — base URL for /artifact/<id> links when clients use another host
#   BAGEL_ARTIFACT_DIR
#   BAGEL_LOAD_MODE=1|2|3  — bf16 | NF4 | INT8 (same as generate_multimodal.py --mode)
#   BAGEL_MAX_LATENT_SIZE=64 — if ema.safetensors latent_pos_embed rows are 4096, must be 64 (see config.json)
#   BAGEL_RESPONSE_PATH_ONLY=1  BAGEL_SHARED_OUTPUT_ROOT=/nfs/...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${BAGEL_REPO:-}" ]]; then
  echo "Error: set BAGEL_REPO to the root of a clone of https://github.com/bytedance-seed/BAGEL" >&2
  exit 1
fi

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/bagel-7b-mot}"

HOST="${BAGEL_SERVER_HOST:-0.0.0.0}"
PORT="${BAGEL_SERVER_PORT:-8875}"
EXTRA=()
if [[ -n "${BAGEL_PUBLIC_BASE_URL:-}" ]]; then
  EXTRA+=(--public-base-url "$BAGEL_PUBLIC_BASE_URL")
fi
if [[ -n "${BAGEL_ARTIFACT_DIR:-}" ]]; then
  EXTRA+=(--artifact-dir "$BAGEL_ARTIFACT_DIR")
fi
if [[ -n "${BAGEL_SHARED_OUTPUT_ROOT:-}" ]]; then
  EXTRA+=(--shared-output-root "$BAGEL_SHARED_OUTPUT_ROOT")
fi
if [[ "${BAGEL_RESPONSE_PATH_ONLY:-}" =~ ^(1|true|yes)$ ]]; then
  EXTRA+=(--response-path-only)
fi

MODE="${BAGEL_LOAD_MODE:-1}"

exec python inference/bagel_generation_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --bagel-repo "$BAGEL_REPO" \
  --model-path "$BAGEL_MODEL_PATH" \
  --mode "$MODE" \
  "${EXTRA[@]}" \
  "$@"
