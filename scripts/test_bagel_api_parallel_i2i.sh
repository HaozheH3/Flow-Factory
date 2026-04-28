#!/usr/bin/env bash
# =============================================================================
# Parallel I2I load test — BAGEL ``POST /api`` with **local paths only**
# =============================================================================
#
# Same discovery rules as ``test_klein_api_parallel_i2i.sh``: read
# ``generation_params.json``, resolve ``reference_images`` URLs via
# ``reference_selection*.json`` (``local_path`` must exist on this machine so the
# test client can POST those paths; the BAGEL server must be able to read the same
# paths — typically shared NFS or the same host).
#
# ``output_path`` is always a local path on the server (see ``BAGEL_TEST_OUTPUT_DIR``).
#
# Usage
# -----
#   export BAGEL_REPO=/path/to/BAGEL   # only needed when starting the server, not this script
#   bash scripts/test_bagel_api_parallel_i2i.sh
#   (IP and BAGEL_TEST_OUTPUT_DIR are exported below; edit for your host/paths)
#
# Optional env
# ------------
#   PORT=8875
#   N_REQUESTS=64
#   CONCURRENCY=8
#   TIMEOUT=900
#   BAGEL_TEST_OUTPUT_DIR=/path/on/server
#   BAGEL_NUM_INFERENCE_STEPS=50   → --num-inference-steps (POST body field)
#   BAGEL_NO_OUTPUT_PATH=1
#   NO_PROGRESS=1  → --no-progress
#   EXAMPLES_DIR=/path/to/production_searchbetter_top500_sft_qw1_debug
#
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export BAGEL_NUM_INFERENCE_STEPS=30
export BAGEL_TEST_OUTPUT_DIR="${BAGEL_TEST_OUTPUT_DIR:-/primus_xpfs_workspace_T04/haozhe/Flow-Factory/scripts/test_bagel_server_outputs}"
export IP="${IP:-33.3.168.168}"
export PORT="${PORT:-9080}"

EXAMPLES_DIR="${EXAMPLES_DIR:-/primus_xpfs_workspace_T04/haozhe/ToolGen/phase4_agent/production_searchbetter_top500_sft_qw1_debug}"

EXTRA=()
[[ "${BAGEL_NO_OUTPUT_PATH:-}" =~ ^(1|true|yes)$ ]] && EXTRA+=(--no-output-path)
[[ "${NO_PROGRESS:-}" =~ ^(1|true|yes)$ ]] && EXTRA+=(--no-progress)
[[ -n "${BAGEL_NUM_INFERENCE_STEPS:-}" ]] && EXTRA+=(--num-inference-steps "${BAGEL_NUM_INFERENCE_STEPS}")

exec python3 "${ROOT}/scripts/test_bagel_parallel_i2i_requests.py" "$IP" \
  --port "${PORT}" \
  --examples-dir "${EXAMPLES_DIR}" \
  -n "${N_REQUESTS:-64}" \
  -j "${CONCURRENCY:-16}" \
  --timeout "${TIMEOUT:-900}" \
  "${EXTRA[@]}"
