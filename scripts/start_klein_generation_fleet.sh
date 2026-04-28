#!/usr/bin/env bash
# Start one Klein HTTP worker per GPU on this machine (recommended layout).
#
# Design
# -----
# - One OS process per GPU, each with CUDA_VISIBLE_DEVICES set to a single card
#   and its own port. Each process loads one Flux2KleinPipeline — avoids VRAM
#   duplication and CUDA context fights from sharing one GPU across workers.
# - "n_worker per GPU" for a single large diffusion model usually means *queue depth*,
#   not extra Python replicas: extra workers on the same GPU only contend for the
#   same memory and kernels. Prefer scaling out to more GPUs or more machines.
# - Load balancing:
#     * Client-side (ToolGen): set KLEIN_GEN_BASE_URL to a comma-separated list of
#       worker base URLs; generation_apis round-robins (or KLEIN_LB_STRATEGY=random).
#     * Or put nginx in front (see scripts/nginx_klein_gateway.example.conf): workers
#       bind 127.0.0.1 only, e.g. KLEIN_FLEET_BASE_PORT=18765, then one VIP
#       http://NODE:8080/api → least_conn to workers. ToolGen:
#         export KLEIN_GEN_BASE_URL=http://NODE:8080/api
#
# Shared NFS, no PNG bytes in POST responses: on each worker set
#   export KLEIN_RESPONSE_PATH_ONLY=1
#   export KLEIN_SHARED_OUTPUT_ROOT=/xpfs/.../klein_out   # optional if every request sends output_path
# and start workers with those env vars (this script forwards them as flags).
#
# After this script prints KLEIN_GEN_BASE_URL, export it on machines running ToolGen.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/black-forest-labs/FLUX.2-klein-base-9B}"
BASE_PORT="${KLEIN_FLEET_BASE_PORT:-8765}"
# Override GPU count (default: nvidia-smi -L)
N_GPU="${N_GPU:-}"

if [[ -z "${N_GPU}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  N_GPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
N_GPU="${N_GPU:-1}"

BIND_HOST="${KLEIN_FLEET_BIND_HOST:-0.0.0.0}"
# Hostname clients use inside artifact URLs (GET). Default loopback; set to hostname/IP clients can reach.
PUBLIC_HOST="${KLEIN_FLEET_PUBLIC_HOST:-127.0.0.1}"

PIDS=()
cleanup() {
  echo "Stopping fleet workers..."
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

EXTRA=()
[[ -n "${KLEIN_ARTIFACT_DIR:-}" ]] && EXTRA+=(--artifact-dir "$KLEIN_ARTIFACT_DIR")
[[ -n "${CPU_OFFLOAD:-}" ]] && EXTRA+=(--cpu-offload)
[[ -n "${COMPILE:-}" ]] && EXTRA+=(--compile)
if [[ "${KLEIN_RESPONSE_PATH_ONLY:-}" =~ ^(1|true|yes)$ ]]; then
  EXTRA+=(--response-path-only)
fi
if [[ -n "${KLEIN_SHARED_OUTPUT_ROOT:-}" ]]; then
  EXTRA+=(--shared-output-root "$KLEIN_SHARED_OUTPUT_ROOT")
fi

for ((i = 0; i < N_GPU; i++)); do
  port=$((BASE_PORT + i))
  pub="http://${PUBLIC_HOST}:${port}"
  echo "Starting worker gpu=${i} port=${port} public_base=${pub}"
  CUDA_VISIBLE_DEVICES="${i}" python inference/klein_generation_server.py \
    --host "$BIND_HOST" \
    --port "$port" \
    --model-path "$MODEL_PATH" \
    --public-base-url "$pub" \
    "${EXTRA[@]}" &
  PIDS+=($!)
done

URLS=""
for ((i = 0; i < N_GPU; i++)); do
  port=$((BASE_PORT + i))
  URLS+="http://${PUBLIC_HOST}:${port},"
done
URLS="${URLS%,}"

echo ""
echo "Fleet running (${N_GPU} workers). For ToolGen on a host that can reach ${PUBLIC_HOST}:"
echo "  export KLEIN_GEN_BASE_URL=${URLS}"
echo "Optional load spread:  export KLEIN_LB_STRATEGY=random"
echo "Single nginx entrypoint (see scripts/nginx_klein_gateway.example.conf):"
echo "  export KLEIN_GEN_BASE_URL=http://${PUBLIC_HOST}:8080/api"
echo "Stop: Ctrl+C (or kill PIDs above)"

wait
