#!/usr/bin/env bash
# Run unit tests + AA synth eval-metadata schema check + optional ToolGen standalone prepare.
#
# Usage:
#   bash scripts/run_toolgen_judge_stack_tests.sh
#   AA_SYNTH_METADATA_JSONL=/path/to.jsonl bash scripts/run_toolgen_judge_stack_tests.sh
#   RUN_TASK_C_PREPARE=1 bash scripts/run_toolgen_judge_stack_tests.sh
#
# Environment:
#   FLOW_FACTORY_ROOT   repo root (default: parent of this script's directory)
#   TOOLGEN_ROOT        ToolGen checkout (default: sibling ../ToolGen next to repo name not assumed — set explicitly)
#   AA_SYNTH_METADATA_JSONL  metadata JSONL (default: ToolGen path below)
#   SCHEMA_MAX_ROWS     0 = full file (default: 0)
#   RUN_TASK_C_PREPARE  set to 1 to also run standalone judge prepare on a task-C vllm jsonl
#   TASK_C_JSONL        default: ToolGen phase6 vllm_responses_run01.jsonl

set -euo pipefail

FLOW_FACTORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FLOW_FACTORY_ROOT}"

export PYTHONPATH="${FLOW_FACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

TOOLGEN_ROOT="${TOOLGEN_ROOT:-/primus_xpfs_workspace_T04/haozhe/ToolGen}"
AA_SYNTH_METADATA_JSONL="${AA_SYNTH_METADATA_JSONL:-${TOOLGEN_ROOT}/phase2_prompt_generation/AA_synth_all_prompts_metadata_eval.jsonl}"
SCHEMA_MAX_ROWS="${SCHEMA_MAX_ROWS:-0}"
TASK_C_JSONL="${TASK_C_JSONL:-${TOOLGEN_ROOT}/phase6_sft/outputs/vllm_responses_run01.jsonl}"
STANDALONE_OUT="${STANDALONE_OUT:-/tmp/toolgen_judge_stack_test_out}"

echo "== pytest: toolgen searchbetter judge =="
python -m pytest tests/test_toolgen_searchbetter_judge_reward.py -q

echo "== schema: AA synth eval metadata (${AA_SYNTH_METADATA_JSONL}) =="
if [[ ! -f "${AA_SYNTH_METADATA_JSONL}" ]]; then
  echo "ERROR: AA_SYNTH_METADATA_JSONL not found: ${AA_SYNTH_METADATA_JSONL}" >&2
  exit 1
fi
schema_args=(--jsonl "${AA_SYNTH_METADATA_JSONL}" --smoke-flow-prompt --check-run-dirs)
if [[ "${SCHEMA_MAX_ROWS}" != "0" ]]; then
  schema_args+=(--max-rows "${SCHEMA_MAX_ROWS}")
fi
python scripts/check_aa_synth_eval_metadata_schema.py "${schema_args[@]}"

if [[ "${RUN_TASK_C_PREPARE:-0}" == "1" ]]; then
  echo "== standalone prepare: task-c jsonl (${TASK_C_JSONL}) =="
  if [[ ! -f "${TASK_C_JSONL}" ]]; then
    echo "ERROR: TASK_C_JSONL not found: ${TASK_C_JSONL}" >&2
    exit 1
  fi
  mkdir -p "${STANDALONE_OUT}"
  python scripts/run_toolgen_searchbetter_judge_standalone_eval.py \
    --mode prepare \
    --task-c-jsonl "${TASK_C_JSONL}" \
    --output-dir "${STANDALONE_OUT}" \
    --toolgen-root "${TOOLGEN_ROOT}"
  echo "Wrote: ${STANDALONE_OUT}/task_c_prepare_user_text_prompt.txt"
fi

echo "== all steps passed =="
