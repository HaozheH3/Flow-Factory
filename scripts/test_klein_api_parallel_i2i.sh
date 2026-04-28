#!/usr/bin/env bash
# =============================================================================
# Parallel I2I load test — Klein ``POST /api`` with **local paths only**
# =============================================================================
#
# Klein does **not** fetch ``http(s)`` URLs in ``multi_modal_data``. Each condition
# image must be an **absolute local path** readable on the inference server.
#
# This script reads ``generation_params.json`` (``prompt`` + ``reference_images``).
# When ``reference_images`` are URLs, paths are resolved via ``reference_selection*.json``
# in the same ``results/<run_id>/`` folder (``candidate_image_mappings`` / ``selected_image``:
# ``url`` → ``local_path``). If any URL cannot be resolved to an **existing** file,
# that example is skipped (the Python runner needs 64 resolvable examples by default).
#
# ``output_path`` is always a local path on the server (see ``KLEIN_TEST_OUTPUT_DIR``).
#
# Usage
# -----
#   bash scripts/test_klein_api_parallel_i2i.sh
#   (IP and KLEIN_TEST_OUTPUT_DIR are exported below; edit those lines to point at your host/paths)
#
# Optional env
# ------------
#   PORT=8080
#   N_REQUESTS=64
#   CONCURRENCY=8
#   TIMEOUT=900
#   KLEIN_TEST_OUTPUT_DIR=/path/on/server
#   KLEIN_NO_OUTPUT_PATH=1
#   NO_PROGRESS=1  → --no-progress (no tqdm bar; consistency block printed after all finish)
#   EXAMPLES_DIR=/path/to/production_searchbetter_top500_sft_qw1_debug
#
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export KLEIN_TEST_OUTPUT_DIR=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/scripts/test_klein_server_outputs
export IP=33.3.168.29
export PORT="${PORT:-8080}"

EXAMPLES_DIR="${EXAMPLES_DIR:-/primus_xpfs_workspace_T04/haozhe/ToolGen/phase4_agent/production_searchbetter_top500_sft_qw1_debug}"

EXTRA=()
[[ "${KLEIN_NO_OUTPUT_PATH:-}" =~ ^(1|true|yes)$ ]] && EXTRA+=(--no-output-path)
[[ "${NO_PROGRESS:-}" =~ ^(1|true|yes)$ ]] && EXTRA+=(--no-progress)

exec python3 "${ROOT}/scripts/test_klein_parallel_i2i_requests.py" "$IP" \
  --port "${PORT}" \
  --examples-dir "${EXAMPLES_DIR}" \
  -n "${N_REQUESTS:-64}" \
  -j "${CONCURRENCY:-16}" \
  --timeout "${TIMEOUT:-900}" \
  "${EXTRA[@]}"
