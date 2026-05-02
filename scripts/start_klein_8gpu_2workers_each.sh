#!/usr/bin/env bash
# Start 8 GPUs × 2 workers = 16 Klein HTTP workers on loopback, plus one round-robin
# gateway on 0.0.0.0:${KLEIN_GATEWAY_PORT:-8080} so clients use a single URL:
#   http://<NODE_IP>:8080/api
#
# VRAM: two full Klein pipelines per GPU doubles memory use vs one worker per GPU;
# reduce WORKERS_PER_GPU to 1 if you hit OOM.
#
# Env (optional):
#   N_GPU=8 WORKERS_PER_GPU=2
#   KLEIN_GPU_IDS=6,7   comma-separated physical GPU ids (if set, N_GPU is inferred from the list)
#   KLEIN_FLEET_BASE_PORT=18765   first worker port (…+15 for 16 workers)
#   KLEIN_GATEWAY_PORT=8080
#   KLEIN_EXTERNAL_IP / NODE_IP  hostname/IP embedded in rewritten artifact URLs (default: first `hostname -I` or 127.0.0.1)
#   MODEL_PATH, KLEIN_RESPONSE_PATH_ONLY, KLEIN_SHARED_OUTPUT_ROOT, CPU_OFFLOAD, COMPILE, KLEIN_ARTIFACT_DIR
#   KLEIN_STARTUP_SLEEP=45  seconds to wait after spawning workers (ASCII progress bar during wait)
set -xeuo pipefail
# conda activate /primus_xpfs_workspace_T04/haozhe/flow_env
cd /primus_xpfs_workspace_T04/haozhe/Flow-Factory
# export KLEIN_EXTERNAL_IP=33.3.181.41
# export KLEIN_EXTERNAL_IP=33.3.187.171
export KLEIN_EXTERNAL_IP=33.3.188.228
# export CUDA_VISIBLE_DEVICES=4,5
export KLEIN_GPU_IDS=4,5,6,7
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/primus_xpfs_workspace_T04/haozhe/gen_models/black-forest-labs/flux2-klein-4b-base}"
N_GPU="${N_GPU:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
BASE_PORT="${KLEIN_FLEET_BASE_PORT:-18795}"
GATEWAY_PORT="${KLEIN_GATEWAY_PORT:-8082}"

if [[ -n "${KLEIN_GPU_IDS:-}" ]]; then
  IFS=',' read -ra KLEIN_GPU_DEVICE_LIST <<< "${KLEIN_GPU_IDS// /}"
  N_GPU=${#KLEIN_GPU_DEVICE_LIST[@]}
else
  KLEIN_GPU_DEVICE_LIST=()
  for ((_i = 0; _i < N_GPU; _i++)); do
    KLEIN_GPU_DEVICE_LIST+=("${_i}")
  done
fi

_ext="${KLEIN_EXTERNAL_IP:-${NODE_IP:-}}"
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
[[ -n "${KLEIN_ARTIFACT_DIR:-}" ]] && EXTRA+=(--artifact-dir "$KLEIN_ARTIFACT_DIR")
[[ -n "${CPU_OFFLOAD:-}" ]] && EXTRA+=(--cpu-offload)
[[ -n "${COMPILE:-}" ]] && EXTRA+=(--compile)
if [[ "${KLEIN_RESPONSE_PATH_ONLY:-}" =~ ^(1|true|yes)$ ]]; then
  EXTRA+=(--response-path-only)
fi
if [[ -n "${KLEIN_SHARED_OUTPUT_ROOT:-}" ]]; then
  EXTRA+=(--shared-output-root "$KLEIN_SHARED_OUTPUT_ROOT")
fi

BACKENDS=""
echo "Starting ${N_GPU} GPUs × ${WORKERS_PER_GPU} workers (base port ${BASE_PORT})..."
for ((g = 0; g < N_GPU; g++)); do
  for ((w = 0; w < WORKERS_PER_GPU; w++)); do
    port=$((BASE_PORT + g * WORKERS_PER_GPU + w))
    BACKENDS+="127.0.0.1:${port},"
    echo "  GPU ${KLEIN_GPU_DEVICE_LIST[g]} worker ${w} → 127.0.0.1:${port}"
    CUDA_VISIBLE_DEVICES="${KLEIN_GPU_DEVICE_LIST[g]}" python inference/klein_generation_server.py \
      --host 127.0.0.1 \
      --port "${port}" \
      --model-path "${MODEL_PATH}" \
      --public-base-url "http://127.0.0.1:${port}" \
      "${EXTRA[@]}" &
    WORKER_PIDS+=($!)
  done
done
BACKENDS="${BACKENDS%,}"

wait_secs="${KLEIN_STARTUP_SLEEP:-45}"
echo "Waiting up to ${wait_secs}s for workers to load (set KLEIN_STARTUP_SLEEP to change)..."
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
  --external-host "${EXTERNAL_IP}" &
GATEWAY_PID=$!

echo ""
echo "Clients / tests use:"
echo "  http://${EXTERNAL_IP}:${GATEWAY_PORT}/api"
echo ""
echo "Run load test from another host:"
echo "  bash scripts/test_klein_api_parallel.sh ${EXTERNAL_IP}"
echo "Stop: Ctrl+C"

wait "${GATEWAY_PID}"
