#!/usr/bin/env bash
# =============================================================================
# Parallel POST load test — Klein HTTP generator (gateway or worker)
# =============================================================================
#
# What “compatible” means here
# ----------------------------
# Integration with ToolGen is defined by **request JSON** and **response body**
# matching ``ToolGen/phase4_agent/generation_apis.py`` (``single_call_t2i`` /
# ``single_call_i2i``): same keys, same ``data: {...}`` SSE-style line in the
# response, same parsing of ``code`` and ``data.choices[0].message.content``.
# Other HTTP details (paths, gateway vs direct worker) are flexible as long as
# POST + JSON + that response shape work.
#
# Caveat: optional ``output_path`` on the request (Klein extension)
# -----------------------------------------------------------------
# The upstream Youliao-style API does **not** require a destination path; Klein
# adds an **optional** JSON field ``output_path`` (string, filesystem path on
# the **server** that runs inference, typically on **shared NFS** visible to all
# workers and to ToolGen).
#
#   • If you omit ``output_path``: the server still returns success; ``content``
#     is usually an ``http(s)`` URL to ``GET`` the PNG (artifact or gateway URL).
#   • If you set ``output_path``: the worker **also** writes the PNG to that path
#     (parents created). ToolGen can pass this via ``single_call_*(...,
#     output_path=...)`` when using Klein backends.
#
# This test driver (Python) sends **64 diverse prompts** by default and, unless
# disabled, a **meaningful** ``output_path`` per request:
#   ``<base>/t2i_{idx:02d}_{slug_from_prompt}.png``
# where ``<base>`` is ``KLEIN_TEST_OUTPUT_DIR`` if set, else
#   ``<repo>/scripts/test_klein_server_outputs/run_<timestamp>/``
# That directory is created locally before requests; on a cluster, set
# ``KLEIN_TEST_OUTPUT_DIR`` to a path that **exists on the inference server** (NFS).
#
# To run **without** ``output_path`` in JSON (artifact URL only):
#   KLEIN_NO_OUTPUT_PATH=1 bash scripts/test_klein_api_parallel.sh <ip>
#
# Usage (only IP required)
# ------------------------
#   bash scripts/test_klein_api_parallel.sh 10.0.0.5
#
# Optional env
# ------------
#   PORT=8080          gateway port (default 8080)
#   N_REQUESTS=64      total POSTs (default 64; capped at 64 built-in prompts)
#   CONCURRENCY=16     max in-flight (default 16)
#   TIMEOUT=600        seconds per request
#   KLEIN_TEST_OUTPUT_DIR=/path/on/shared/fs   # optional; overrides default run dir
#   KLEIN_NO_OUTPUT_PATH=1                     # omit output_path from every request
#
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# IP="${1:?Usage: $0 <server_ip>   # optional env: PORT N_REQUESTS CONCURRENCY TIMEOUT KLEIN_TEST_OUTPUT_DIR KLEIN_NO_OUTPUT_PATH}"
export PORT="${PORT:-8080}"

EXTRA=()
if [[ "${KLEIN_NO_OUTPUT_PATH:-}" =~ ^(1|true|yes)$ ]]; then
  EXTRA+=(--no-output-path)
fi
export KLEIN_TEST_OUTPUT_DIR=/primus_xpfs_workspace_T04/haozhe/Flow-Factory/scripts/test_klein_server_outputs
export IP=33.3.168.29
exec python3 "${ROOT}/scripts/test_klein_parallel_requests.py" "${IP}" \
  --port "${PORT}" \
  -n "${N_REQUESTS:-64}" \
  -j "${CONCURRENCY:-16}" \
  --timeout "${TIMEOUT:-600}" \
  "${EXTRA[@]}"
