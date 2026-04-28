#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone ToolGen SearchBetter judge: Frontier API + ToolGen evaluation protocol.

Two input styles:

1) **Phase5 hard eval** — ``--hard-jsonl`` + ``--results-dir`` (phase4 scan root), same as
   ``evaluate_searchbetter_hard_direct.build_rows_for_evaluation``.

2) **Phase6 Task C row** — ``--task-c-jsonl`` only (e.g. ``phase6_sft/outputs/vllm_responses_run01.jsonl``).
   Resolution matches ``phase6_sft/reward_label_task_c.py``: ``collection.phase5_folder`` →
   ``evaluation_dataset.jsonl`` + phase4 scan root from ``raw_generation_folder``. Checklist/rubric
   still come from ``provenance.request_dataset_path`` + ``request_dataset_index`` via
   ``load_request_meta`` (same as Task C; not duplicated inside the vllm row unless you add them).

Modes:
  eval (default) — ``process_single_row`` (Frontier API).
  prepare — ``prepare_evaluate_one_judge_multimodal`` for the augmented variant; writes
  ``task_c_prepare_user_text_prompt.txt`` (no API).
  compare-prompt — diff rebuilt ``user_text_prompt`` vs ``--reference-prompt-txt`` (no API).

Environment: same as ToolGen ``frontier_model.LLMClient`` (``LLM_TOKEN`` / ``FRONTIER_LLM_TOKEN``,
``FRONTIER_LLM_API_URL``, etc.).

Example (evaluate one row):
  python scripts/run_toolgen_searchbetter_judge_standalone_eval.py \\
    --hard-jsonl /primus_xpfs_workspace_T04/haozhe/ToolGen/phase5_data_collection/searchbetter_evaldataset_hard.jsonl \\
    --results-dir /primus_xpfs_workspace_T04/haozhe/ToolGen/phase4_agent/visual_rerun_klein_sft_qw2_searchset_hard \\
    --output-dir /tmp/toolgen_judge_standalone_out \\
    --model doubao-seed-2.0-mini \\
    --limit 1

Example (phase6 Task C vllm row only — rebuild judge text, no API):
  python scripts/run_toolgen_searchbetter_judge_standalone_eval.py \\
    --mode prepare \\
    --task-c-jsonl /primus_xpfs_workspace_T04/haozhe/ToolGen/phase6_sft/outputs/vllm_responses_run01.jsonl

Example (prompt parity vs a saved evaluation entry):
  ``searchbetter_evaldataset_hard.jsonl`` uses sparse ``request_index`` (not 1..N). Select with
  ``--trajectory-id`` substring, e.g. the row that shares the same ``user_prompt`` as the saved entry:
  python scripts/run_toolgen_searchbetter_judge_standalone_eval.py \\
    --mode compare-prompt \\
    --hard-jsonl .../searchbetter_evaldataset_hard.jsonl \\
    --results-dir .../visual_rerun_klein_sft_qw2_searchset_hard \\
    --reference-prompt-txt .../entries/0001_request_0001__.../augmented_prompt_context.txt \\
    --trajectory-id request_0468

  Exact byte match to a reference ``augmented_prompt_context.txt`` requires the same
  ``trajectory_id`` / ``request_meta`` as that run; otherwise expect diffs only in metadata blocks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _ensure_toolgen_paths(toolgen_root: Path) -> None:
    p5 = str((toolgen_root / "phase5_data_collection").resolve())
    p4 = str((toolgen_root / "phase4_agent").resolve())
    for p in (p5, p4):
        if p not in sys.path:
            sys.path.insert(0, p)


def _normalize_prompt_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip() + "\n"


def _ensure_phase6_path(toolgen_root: Path) -> None:
    p6 = str((toolgen_root / "phase6_sft").resolve())
    if p6 not in sys.path:
        sys.path.insert(0, p6)


def _select_task_c_row(
    rows: List[Dict[str, Any]],
    *,
    corpus_line: Optional[int],
    inference_id: Optional[str],
    subtask: str = "C",
) -> Dict[str, Any]:
    if inference_id:
        for r in rows:
            if str(r.get("inference_id") or "") == inference_id:
                return r
        raise SystemExit(f"No row with inference_id={inference_id!r} in task-c jsonl")
    if corpus_line is not None:
        for r in rows:
            if r.get("corpus_line") != corpus_line:
                continue
            if subtask and r.get("subtask") != subtask:
                continue
            return r
        raise SystemExit(
            f"No row with corpus_line={corpus_line!r} and subtask={subtask!r} in task-c jsonl"
        )
    for r in rows:
        if r.get("subtask") == subtask:
            return r
    raise SystemExit(f"No row with subtask={subtask!r} in task-c jsonl")


def _merge_provenance_into_judge_row(judge_row: Dict[str, Any], task_c_row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judge_row)
    prov = task_c_row.get("provenance") or {}
    ds_path = prov.get("request_dataset_path") or out.get("request_dataset_path")
    ds_index = prov.get("request_dataset_index")
    if ds_index is None:
        ds_index = out.get("request_dataset_index")
    if ds_path is not None:
        out["request_dataset_path"] = ds_path
    if ds_index is not None:
        out["request_dataset_index"] = ds_index
    return out


def _resolve_judge_row_from_task_c(
    toolgen_root: Path,
    task_c_row: Dict[str, Any],
    ev: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _ensure_phase6_path(toolgen_root)
    import reward_label_task_c as rlc  # type: ignore[import-not-found]

    trajectory_id = str(task_c_row.get("trajectory_id") or "").strip()
    if not trajectory_id:
        raise SystemExit("task-c row missing trajectory_id")

    collection = task_c_row.get("collection") or {}
    phase5_folder = str((collection or {}).get("phase5_folder") or "").strip()
    if not phase5_folder:
        raise SystemExit("task-c row missing collection.phase5_folder")

    phase5_dir = rlc.resolve_phase5_dir(phase5_folder)
    eval_row_disk = rlc.load_evaluation_row(phase5_dir, trajectory_id)
    if eval_row_disk is None:
        raise SystemExit(
            f"trajectory_id {trajectory_id!r} not found in {phase5_dir / 'evaluation_dataset.jsonl'}"
        )

    judge_row, meta = rlc.resolve_judge_row_direct_eval_protocol(
        ev,
        eval_row_disk=eval_row_disk,
        trajectory_id=trajectory_id,
        hard_dataset_path=None,
    )
    return _merge_provenance_into_judge_row(judge_row, task_c_row), meta


def _build_prepare_payload_for_row(
    row: Dict[str, Any],
    ev: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta_cache: Dict[str, Any] = {}
    request_meta = ev.load_request_meta(
        request_dataset_path=row.get("request_dataset_path"),
        request_dataset_index=row.get("request_dataset_index"),
        cache=meta_cache,
    )
    if request_meta is None:
        request_meta = {}
    visual_context = ev.build_visual_reference_context(row)
    payload, prep_err = ev.prepare_evaluate_one_judge_multimodal(
        row=row,
        request_meta=request_meta,
        visual_context=visual_context,
        variant="augmented",
        image_url=row.get("augmented_generation_url")
        if isinstance(row.get("augmented_generation_url"), str)
        else None,
        target_image_local_path=None,
    )
    if prep_err is not None:
        raise SystemExit(f"prepare_evaluate_one_judge_multimodal failed: {prep_err}")
    if payload is None:
        raise SystemExit("prepare_evaluate_one_judge_multimodal returned no payload")
    return payload, request_meta


def _pick_matched_row(
    rows: List[Dict[str, Any]],
    *,
    request_index: Optional[int],
    trajectory_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("match_status") != "matched":
            continue
        if request_index is not None and r.get("request_index") != request_index:
            continue
        if trajectory_id is not None:
            tid = str(r.get("trajectory_id") or "")
            if trajectory_id not in tid:
                continue
        return r
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--toolgen-root",
        type=Path,
        default=Path("/primus_xpfs_workspace_T04/haozhe/ToolGen"),
        help="Path to ToolGen repository root",
    )
    parser.add_argument(
        "--hard-jsonl",
        type=Path,
        default=None,
        help="Phase5 hard dataset JSONL (requires --results-dir unless using --task-c-jsonl)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Phase4 scan root for build_rows_for_evaluation (requires --hard-jsonl unless using --task-c-jsonl)",
    )
    parser.add_argument(
        "--task-c-jsonl",
        type=Path,
        default=None,
        help="Phase6 vllm / Task C JSONL; derives phase5 evaluation row + phase4 root from the row (no --hard-jsonl)",
    )
    parser.add_argument(
        "--task-c-corpus-line",
        type=int,
        default=None,
        help="Select task-c row by corpus_line (with --task-c-subtask)",
    )
    parser.add_argument(
        "--task-c-inference-id",
        type=str,
        default=None,
        help="Select task-c row by exact inference_id",
    )
    parser.add_argument(
        "--task-c-subtask",
        type=str,
        default="C",
        help="Default subtask filter when picking first matching row (default: C)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/toolgen_judge_standalone_out"))
    parser.add_argument("--model", type=str, default="doubao-seed-2.0-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1, help="Max rows to run in eval mode after filtering")
    parser.add_argument(
        "--mode",
        choices=("eval", "prepare", "compare-prompt"),
        default="eval",
        help="eval: Frontier; prepare: build augmented judge text only; compare-prompt: diff vs reference file",
    )
    parser.add_argument(
        "--reference-prompt-txt",
        type=Path,
        default=None,
        help="Saved augmented_prompt_context.txt for compare-prompt mode",
    )
    parser.add_argument(
        "--request-index",
        type=int,
        default=None,
        help="Select resolved row by request_index (compare-prompt or single-row debug)",
    )
    parser.add_argument(
        "--trajectory-id",
        type=str,
        default=None,
        help="Select resolved row by trajectory_id substring (matched rows only)",
    )
    args = parser.parse_args()

    toolgen_root = args.toolgen_root.resolve()
    _ensure_toolgen_paths(toolgen_root)

    import evaluate_searchbetter_hard_direct as ev  # type: ignore[import-not-found]

    from evaluate_searchbetter_hard_direct import (  # type: ignore[import-not-found]
        build_rows_for_evaluation,
        process_single_row,
        read_jsonl,
    )

    use_task_c = args.task_c_jsonl is not None
    has_hard_jsonl = args.hard_jsonl is not None
    has_results_dir = args.results_dir is not None
    if has_hard_jsonl ^ has_results_dir:
        raise SystemExit("Use --hard-jsonl and --results-dir together (both required for the hard-eval path)")
    use_hard = has_hard_jsonl and has_results_dir
    if use_task_c and use_hard:
        raise SystemExit("Do not combine --task-c-jsonl with --hard-jsonl / --results-dir")
    if not use_task_c and not use_hard:
        raise SystemExit("Provide --task-c-jsonl OR both --hard-jsonl and --results-dir")

    task_c_row: Optional[Dict[str, Any]] = None
    judge_resolution: Dict[str, Any] = {}
    row: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = []

    if use_task_c:
        tc_path = Path(args.task_c_jsonl).resolve()
        tc_rows = read_jsonl(tc_path)
        if not tc_rows:
            raise SystemExit(f"No rows in {tc_path}")
        task_c_row = _select_task_c_row(
            tc_rows,
            corpus_line=args.task_c_corpus_line,
            inference_id=args.task_c_inference_id,
            subtask=args.task_c_subtask,
        )
        row, judge_resolution = _resolve_judge_row_from_task_c(toolgen_root, task_c_row, ev)
        print(f"task_c judge resolution: {judge_resolution}")
        print(
            f"selected inference_id={task_c_row.get('inference_id')!r} "
            f"trajectory_id={row.get('trajectory_id')!r} match_status={row.get('match_status')!r}"
        )
    else:
        hard_rows = read_jsonl(Path(args.hard_jsonl).resolve())
        if not hard_rows:
            raise SystemExit(f"No rows in {args.hard_jsonl}")
        rows, stats = build_rows_for_evaluation(hard_rows, Path(args.results_dir).resolve())
        print(f"build_rows_for_evaluation stats: {stats}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare-prompt":
        if args.reference_prompt_txt is None or not args.reference_prompt_txt.is_file():
            raise SystemExit("compare-prompt requires existing --reference-prompt-txt")
        if use_hard:
            if args.request_index is None and args.trajectory_id is None:
                raise SystemExit("compare-prompt (hard path) requires --request-index and/or --trajectory-id")
            row = _pick_matched_row(
                rows,
                request_index=args.request_index,
                trajectory_id=args.trajectory_id,
            )
            if row is None:
                matched = [r for r in rows if r.get("match_status") == "matched"]
                seen: Set[Any] = set()
                sample_idx: List[Any] = []
                for r in matched:
                    ri = r.get("request_index")
                    if ri is None or ri in seen:
                        continue
                    seen.add(ri)
                    sample_idx.append(ri)
                    if len(sample_idx) >= 24:
                        break
                raise SystemExit(
                    "No matched row for the given selectors; "
                    "ensure hard jsonl + results-dir yield match_status=matched. "
                    f"Example request_index values among matched rows (up to 50): {sample_idx!r}"
                )
        else:
            assert row is not None

        ref_bytes = Path(args.reference_prompt_txt).read_text(encoding="utf-8")
        want = _normalize_prompt_text(ref_bytes)

        payload, _request_meta = _build_prepare_payload_for_row(row, ev)
        got = _normalize_prompt_text(str(payload["user_text_prompt"]))
        if got == want:
            print("compare-prompt: OK (exact match after newline normalization).")
            return

        print("compare-prompt: MISMATCH between rebuilt user_text_prompt and reference file.")
        out_path = Path(args.output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "rebuilt_user_text_prompt.txt").write_text(got, encoding="utf-8")
        (out_path / "reference_normalized.txt").write_text(want, encoding="utf-8")
        print(f"Wrote diff inputs to {out_path}/rebuilt_user_text_prompt.txt and reference_normalized.txt")
        raise SystemExit(1)

    if args.mode == "prepare":
        if use_hard:
            if args.request_index is None and args.trajectory_id is None:
                raise SystemExit("prepare (hard path) requires --request-index and/or --trajectory-id")
            row = _pick_matched_row(
                rows,
                request_index=args.request_index,
                trajectory_id=args.trajectory_id,
            )
            if row is None:
                raise SystemExit("prepare: no matched row for selectors")
        assert row is not None

        payload, request_meta = _build_prepare_payload_for_row(row, ev)
        n_check = len(request_meta.get("verification_checklist") or []) if isinstance(request_meta, dict) else 0
        n_rubric = len(request_meta.get("evaluation_rubric") or {}) if isinstance(request_meta, dict) else 0
        out_txt = out_dir / "task_c_prepare_user_text_prompt.txt"
        out_txt.write_text(str(payload["user_text_prompt"]), encoding="utf-8")
        print(
            f"prepare: OK — wrote {out_txt} ({len(str(payload['user_text_prompt']))} chars); "
            f"request_meta checklist items={n_check}, rubric keys={n_rubric}"
        )
        return

    if args.mode != "eval":
        raise SystemExit(f"Unknown mode {args.mode!r}")

    if use_task_c:
        assert row is not None
        res = process_single_row(
            row=row,
            output_dir=out_dir,
            model_name=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            strict_meta_lookup=False,
            prompt_to_entry_dir=None,
        )
        print(
            f"  trajectory_id={row.get('trajectory_id')} request_index={row.get('request_index')} "
            f"meta_ok={res.get('meta_validation_ok')} counters={res.get('counters')}"
        )
        return

    eligible = [
        r
        for r in rows
        if r.get("match_status") == "matched"
        and isinstance(r.get("baseline_generation_url"), str)
        and r.get("baseline_generation_url").strip()
        and isinstance(r.get("augmented_generation_url"), str)
        and r.get("augmented_generation_url").strip()
    ]
    if not eligible:
        raise SystemExit("No eligible matched rows with baseline and augmented URLs")

    subset = eligible[: max(1, args.limit)]
    print(f"Running process_single_row for {len(subset)} row(s) -> {out_dir}")

    for row in subset:
        res = process_single_row(
            row=row,
            output_dir=out_dir,
            model_name=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            strict_meta_lookup=False,
            prompt_to_entry_dir=None,
        )
        print(
            f"  trajectory_id={row.get('trajectory_id')} request_index={row.get('request_index')} "
            f"meta_ok={res.get('meta_validation_ok')} counters={res.get('counters')}"
        )


if __name__ == "__main__":
    main()
