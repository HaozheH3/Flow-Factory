#!/usr/bin/env bash
# Start ToolGen-compatible FLUX.2 Klein HTTP generation server (POST /, same JSON as generation_apis).
#
# ToolGen client (same host as typical dev):
#   export KLEIN_GEN_BASE_URL=http://127.0.0.1:8765
#   # If clients resolve a different host than bind address, set public URL for artifact links:
#   export KLEIN_PUBLIC_BASE_URL=http://127.0.0.1:8765
#
# Multi-GPU on one box: use start_klein_generation_fleet.sh (one worker per GPU) and set
#   KLEIN_GEN_BASE_URL to the comma-separated list it prints (or nginx → single http://ip:port/api).
# Shared NFS, path-only responses (no PNG in POST): KLEIN_RESPONSE_PATH_ONLY=1 and
#   KLEIN_SHARED_OUTPUT_ROOT=/path/on/shared/disk (or pass output_path per request).
#
# Then use adapter ids containing "klein", e.g. --baseline-model-id flux2-klein --augmented-model-id flux2-klein
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MODEL_PATH="${MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/black-forest-labs/FLUX.2-klein-base-9B}"

HOST="${KLEIN_SERVER_HOST:-0.0.0.0}"
PORT="${KLEIN_SERVER_PORT:-8765}"
EXTRA=()
if [[ -n "${KLEIN_PUBLIC_BASE_URL:-}" ]]; then
  EXTRA+=(--public-base-url "$KLEIN_PUBLIC_BASE_URL")
fi
if [[ -n "${KLEIN_ARTIFACT_DIR:-}" ]]; then
  EXTRA+=(--artifact-dir "$KLEIN_ARTIFACT_DIR")
fi
if [[ -n "${CPU_OFFLOAD:-}" ]]; then
  EXTRA+=(--cpu-offload)
fi
if [[ -n "${COMPILE:-}" ]]; then
  EXTRA+=(--compile)
fi

exec python inference/klein_generation_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  "${EXTRA[@]}" \
  "$@"
