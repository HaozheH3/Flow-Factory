#!/usr/bin/env bash
# Start N GPUs × WORKERS_PER_GPU BAGEL HTTP workers on loopback, plus one round-robin
# gateway on 0.0.0.0:${BAGEL_GATEWAY_PORT:-9080} so clients use a single URL:
#   http://<NODE_IP>:9080/api
#
# Same layout as scripts/start_klein_8gpu_2workers_each.sh (uses inference/klein_rr_gateway.py
# with --v1-model-id / --gateway-label for Bagel). For one process on one GPU, use
# scripts/start_bagel_generation_server.sh instead.
#
# VRAM: multiple workers per GPU each load a full BAGEL stack — reduce WORKERS_PER_GPU if OOM.
#
# Env (optional):
#   BAGEL_REPO (required) — clone root of https://github.com/bytedance-seed/BAGEL
#   BAGEL_MODEL_PATH
#   N_GPU=8 WORKERS_PER_GPU=2
#   BAGEL_GPU_IDS=6,7   comma-separated physical GPU ids (if set, N_GPU is inferred from the list)
#   BAGEL_FLEET_BASE_PORT=19765   first worker port
#   BAGEL_GATEWAY_PORT=9080       (default avoids collision with Klein on 8080)
#   BAGEL_EXTERNAL_IP / NODE_IP
#   BAGEL_STARTUP_SLEEP=120     seconds to wait after spawning workers (Bagel load is slow)
#   BAGEL_LOAD_MODE=1|2|3  BAGEL_ARTIFACT_DIR  BAGEL_RESPONSE_PATH_ONLY  BAGEL_SHARED_OUTPUT_ROOT
#   BAGEL_MAX_LATENT_SIZE=64 — must match ema.safetensors (4096 rows => 64)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NODE_IP=33.3.168.168
BAGEL_GPU_IDS=0,1,2,3
BAGEL_REPO=/primus_xpfs_workspace_T04/haozhe/BAGEL
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/bagel-7b-mot}"
export BAGEL_RESPONSE_PATH_ONLY=1 # do not save to a cache dir of the server
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BASE_PORT="${BAGEL_FLEET_BASE_PORT:-19765}"
GATEWAY_PORT="${BAGEL_GATEWAY_PORT:-9080}"


if [[ -z "${BAGEL_REPO:-}" ]]; then
  echo "Error: set BAGEL_REPO to the root of a clone of https://github.com/bytedance-seed/BAGEL" >&2
  exit 1
fi



if [[ -n "${BAGEL_GPU_IDS:-}" ]]; then
  IFS=',' read -ra BAGEL_GPU_DEVICE_LIST <<< "${BAGEL_GPU_IDS// /}"
  N_GPU=${#BAGEL_GPU_DEVICE_LIST[@]}
else
  BAGEL_GPU_DEVICE_LIST=()
  for ((_i = 0; _i < N_GPU; _i++)); do
    BAGEL_GPU_DEVICE_LIST+=("${_i}")
  done
fi

_ext="${BAGEL_EXTERNAL_IP:-${NODE_IP:-}}"
if [[ -z "${_ext}" ]] && command -v hostname >/dev/null 2>&1; then
  _ext="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
EXTERNAL_IP="${_ext:-127.0.0.1}"

WORKER_PIDS=()
cleanup() {
  echo "Stopping gateway and workers..."
  [[ -n "${GATEWAY_PID:-}" ]] && kill "${GATEWAY_PID}" 2>/dev/null || true
  for p in "${WORKER_PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

EXTRA=()
[[ -n "${BAGEL_ARTIFACT_DIR:-}" ]] && EXTRA+=(--artifact-dir "$BAGEL_ARTIFACT_DIR")
if [[ "${BAGEL_RESPONSE_PATH_ONLY:-}" =~ ^(1|true|yes)$ ]]; then
  EXTRA+=(--response-path-only)
fi
if [[ -n "${BAGEL_SHARED_OUTPUT_ROOT:-}" ]]; then
  EXTRA+=(--shared-output-root "$BAGEL_SHARED_OUTPUT_ROOT")
fi

MODE="${BAGEL_LOAD_MODE:-1}"

BACKENDS=""
echo "Starting ${N_GPU} GPUs × ${WORKERS_PER_GPU} BAGEL workers (base port ${BASE_PORT})..."
for ((g = 0; g < N_GPU; g++)); do
  for ((w = 0; w < WORKERS_PER_GPU; w++)); do
    port=$((BASE_PORT + g * WORKERS_PER_GPU + w))
    BACKENDS+="127.0.0.1:${port},"
    echo "  GPU ${BAGEL_GPU_DEVICE_LIST[g]} worker ${w} → 127.0.0.1:${port}"
    CUDA_VISIBLE_DEVICES="${BAGEL_GPU_DEVICE_LIST[g]}" python inference/bagel_generation_server.py \
      --host 127.0.0.1 \
      --port "${port}" \
      --bagel-repo "$BAGEL_REPO" \
      --model-path "$BAGEL_MODEL_PATH" \
      --mode "${MODE}" \
      --public-base-url "http://127.0.0.1:${port}" \
      "${EXTRA[@]}" &
    WORKER_PIDS+=($!)
  done
done
BACKENDS="${BACKENDS%,}"

wait_secs="${BAGEL_STARTUP_SLEEP:-120}"
echo "Waiting up to ${wait_secs}s for workers to load (set BAGEL_STARTUP_SLEEP to change)..."
w=40
for ((i = 1; i <= wait_secs; i++)); do
  n=$((i * w / wait_secs))
  ((n > w)) && n=$w
  printf -v _bar '%*s' "$n" ''
  _bar=${_bar// /#}
  printf -v _rest '%*s' "$((w - n))" ''
  _rest=${_rest// /-}
  printf '\r  [%s%s] %3d/%ds' "$_bar" "$_rest" "$i" "$wait_secs"
  sleep 1
done
printf '\n'

echo "Starting RR gateway on 0.0.0.0:${GATEWAY_PORT} (external artifact host ${EXTERNAL_IP})"
python inference/klein_rr_gateway.py \
  --listen-host 0.0.0.0 \
  --listen-port "${GATEWAY_PORT}" \
  --backends "${BACKENDS}" \
  --external-host "${EXTERNAL_IP}" \
  --v1-model-id bagel-7b-mot \
  --gateway-label bagel-rr-gateway &
GATEWAY_PID=$!

echo ""
echo "Clients / tests use:"
echo "  http://${EXTERNAL_IP}:${GATEWAY_PORT}/api"
echo ""
echo "Stop: Ctrl+C"

wait "${GATEWAY_PID}"
