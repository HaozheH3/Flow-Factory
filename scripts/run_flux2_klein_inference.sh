#!/usr/bin/env bash
# Back-compat wrapper — prefer: ./scripts/run_multimodal_inference.sh flux2-klein
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run_multimodal_inference.sh" flux2-klein "$@"
