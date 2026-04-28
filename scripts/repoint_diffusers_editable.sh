#!/usr/bin/env bash
# Fix ``import diffusers`` resolving to the wrong tree (e.g. Diffuser-Qwen-Image)
# because ``site-packages/__editable__.diffusers-*.pth`` points at that repo's ``src``.
#
# Recommended: let pip regenerate the editable install (updates .pth + dist-info).
#   VENV=/primus_xpfs_workspace_T04/haozhe/flow_env \
#   NEW_DIFFUSERS_REPO=/path/to/huggingface/diffusers \
#   ./scripts/repoint_diffusers_editable.sh
#
# Official clone:
#   git clone https://github.com/huggingface/diffusers.git ~/diffusers
#   (then set NEW_DIFFUSERS_REPO to that directory)
#
# To drop editable entirely and use wheels from PyPI (removes __editable__.diffusers-*.pth):
#   VENV=... ./scripts/repoint_diffusers_editable.sh --pypi
#
# Emergency (not recommended): only rewrite lines in __editable__.diffusers-*.pth
#   VENV=... ./scripts/repoint_diffusers_editable.sh --pth-only /abs/path/to/diffusers/src
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  head -40 "$0" | tail -n +2
  exit 1
}

VENV="${VENV:-${VIRTUAL_ENV:-}}"
if [[ -z "${VENV}" ]]; then
  echo "Set VENV=/path/to/venv or activate your virtualenv (VIRTUAL_ENV)." >&2
  usage
fi

BIN="${VENV}/bin"
if [[ ! -x "${BIN}/python" ]]; then
  echo "No executable: ${BIN}/python" >&2
  exit 1
fi

# Always use ``python -m pip``: many venvs ship a broken ``pip`` script whose shebang
# points at another interpreter, so ``pip install`` mutates the wrong site-packages.
PIP=( "${BIN}/python" -m pip )

SITE="$("${BIN}/python" -c "import site; print([p for p in site.getsitepackages() if p.endswith('site-packages')][0])")"

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  usage
fi

if [[ "${1:-}" == "--pypi" ]]; then
  echo "Uninstalling editable diffusers and installing from PyPI into ${VENV} ..."
  "${PIP[@]}" uninstall -y diffusers || true
  "${PIP[@]}" install "diffusers>=0.36"
  echo "diffusers file:"
  "${BIN}/python" -c "import diffusers; print(diffusers.__file__)"
  exit 0
fi

if [[ "${1:-}" == "--pth-only" ]]; then
  NEW_SRC="${2:?Pass absolute path to the directory that contains the diffusers package (usually .../diffusers/src)}"
  if [[ ! -d "${NEW_SRC}/diffusers" ]]; then
    echo "Expected a directory layout like:  NEW_SRC/diffusers/__init__.py" >&2
    echo "Got NEW_SRC=${NEW_SRC}" >&2
    exit 1
  fi
  shopt -s nullglob
  PTHS=("${SITE}"/__editable__.diffusers-*.pth)
  shopt -u nullglob
  if [[ ${#PTHS[@]} -eq 0 ]]; then
    echo "No ${SITE}/__editable__.diffusers-*.pth found (maybe already using wheels)." >&2
    exit 1
  fi
  for p in "${PTHS[@]}"; do
    echo "Rewriting ${p}"
    printf '%s\n' "$(realpath "${NEW_SRC}")" >"${p}"
  done
  echo "Warning: dist-info direct_url.json may still describe the old repo; prefer reinstall without --pth-only." >&2
  "${BIN}/python" -c "import diffusers; print(diffusers.__file__)"
  exit 0
fi

REPO="${NEW_DIFFUSERS_REPO:-${1:-}}"
if [[ -z "${REPO}" ]]; then
  echo "Set NEW_DIFFUSERS_REPO to a huggingface/diffusers checkout, or pass it as the first argument." >&2
  usage
fi
if [[ ! -f "${REPO}/pyproject.toml" ]] && [[ ! -f "${REPO}/setup.py" ]]; then
  echo "NEW_DIFFUSERS_REPO should be the root of the diffusers git repo (contains pyproject.toml)." >&2
  exit 1
fi

echo "Reinstalling diffusers editable from: ${REPO}"
"${PIP[@]}" uninstall -y diffusers || true
"${PIP[@]}" install -e "${REPO}"
echo "diffusers file:"
"${BIN}/python" -c "import diffusers; print(diffusers.__file__)"
