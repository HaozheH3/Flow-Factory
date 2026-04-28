#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Build Flow-Factory Klein i2i JSONL for NFT + ``toolgen_searchbetter_judge*`` rewards.

Plan
----
1. **Source A — difficulty shards** (``difficulty_label_shards/*.json``): each file is one labeled row
   with ``difficulty_label.difficulty_with_ideal_visual_references`` and the same layout as
   ``collect_prompt_eval_dataset.py`` (``user_prompt``, ``evaluation_context``, ``phase4.run_dir``).

2. **Filter**: keep rows with ``difficulty_with_ideal_visual_references >= --min-visual-difficulty``
   (default 3.5).

3. **Source B — phase4 run** (same roots as ``phase6_sft/run_collect_and_label_prompt_difficulty.sh`` →
   ``scripts/collect_prompt_eval_dataset.py``): require ``generation_params.json`` under ``run_dir``.

4. **Resolve reference images** for ``GeneralDataset`` ``image`` column (loaded as conditioning):
   **First** read ``reference_eval_lookup.json`` under ``run_dir`` when valid (ToolGen
   ``try_load_reference_eval_lookup`` — same pre-resolved ``eval_reference_slots`` as phase5 judge).
   If missing or empty, **rebuild** slots with ``build_eval_reference_slots_from_generation_params``.
   Final fallback: ``run_dir/reference_images/*`` on disk.

5. **Output schema** — ``prompt``, ``image`` (abs paths list), ``user_prompt``, checklist,
   ``evaluation_rubric`` as a **JSON string** (per-row rubric keys differ; a string column avoids
   HuggingFace Arrow struct merge failures in ``GeneralDataset``), ``trajectory_id``,
   ``request_index``,    ``augmented_generation_details`` with ``eval_reference_slots`` (I2I paths from
   ``generation_params.json`` in order, joined to ``analysis`` + ``reference_selection_*.json`` for
   entity/reasoning), etc.

**Parallelism**: default ``--workers`` uses ``min(32, os.cpu_count() or 8)``. Workers run
``_shard_worker_task`` (spawn-safe). Output order follows sorted shard filenames (``Pool.imap``).
Use ``--workers 1`` to disable multiprocessing (still shows tqdm when tqdm is installed).

**Post steps** (same run): optional **train/test split** of the written ``--output`` file; then
**reference image pruning** via ``prune_jsonl_broken_reference_images.py`` (decode check,
``--max-images``, optional pixel percentile) on the resulting JSONL file(s). Web knowledge
``eval_text_knowledge_slots`` is filled from analysis (see ToolGen
``build_eval_text_knowledge_slots_from_analysis``).

Usage
-----
  PYTHONPATH=src python scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py \\
    --shard-dir .../difficulty_label_shards \\
    --output dataset/klein_i2i_hard_visual_35/train.jsonl

Or: ``bash scripts/build_klein_i2i_rl_dataset.sh`` (see that file for ``OUTPUT_JSONL`` / ``WORKERS`` env vars).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


def _ensure_toolgen_phase5_on_path(toolgen_root: Path) -> None:
    p5 = str((toolgen_root / "phase5_data_collection").resolve())
    if p5 not in sys.path:
        sys.path.insert(0, p5)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rubric_dict_only(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k.strip() and isinstance(v, dict):
            out[k.strip()] = dict(v)
    return out


def _summary_used_ref_count(run_dir: Path) -> int:
    s = _read_json(run_dir / "summary.json")
    if not isinstance(s, dict):
        return 0
    ps = s.get("pipeline_summary")
    if not isinstance(ps, dict):
        return 0
    urls = ps.get("reference_image_urls")
    if isinstance(urls, list):
        return len(urls)
    ref_used = ps.get("reference_image_used")
    if isinstance(ref_used, bool) and ref_used:
        return 1
    return 0


def _iter_shard_files(shard_dir: Path) -> Iterator[Path]:
    for p in sorted(shard_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".json":
            yield p


def _list_shard_paths(shard_dir: Path) -> List[Path]:
    return list(_iter_shard_files(shard_dir))


def _load_shard_record(path: Path) -> Optional[Dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, dict):
        return data
    return None


def _paths_from_eval_slots(slots: List[Any]) -> List[str]:
    locals_out: List[str] = []
    seen: set[str] = set()
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        lp = slot.get("reference_local_path")
        if isinstance(lp, str) and lp.strip():
            p = Path(lp.strip())
            if p.is_file():
                abs_p = str(p.resolve())
                if abs_p not in seen:
                    seen.add(abs_p)
                    locals_out.append(abs_p)
                continue
        iu = slot.get("image_url")
        if isinstance(iu, str) and iu.strip():
            p2 = Path(iu.strip())
            if not iu.lower().startswith(("http://", "https://", "data:")) and p2.is_file():
                abs_p = str(p2.resolve())
                if abs_p not in seen:
                    seen.add(abs_p)
                    locals_out.append(abs_p)
    return locals_out


def _visual_difficulty(record: Dict[str, Any]) -> Optional[float]:
    dl = record.get("difficulty_label")
    if not isinstance(dl, dict):
        return None
    v = dl.get("difficulty_with_ideal_visual_references")
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _resolve_local_reference_paths(
    run_dir: Path,
    ev: Any,
) -> Tuple[List[str], Optional[str], str]:
    lookup_slots = ev.try_load_reference_eval_lookup(run_dir, force_refresh=False)
    if isinstance(lookup_slots, list) and lookup_slots:
        from_lookup = _paths_from_eval_slots(lookup_slots)
        if from_lookup:
            return from_lookup, None, "reference_eval_lookup"

    gp = ev.load_generation_params_json(run_dir)
    if not isinstance(gp, dict):
        return [], "missing_or_invalid_generation_params.json", "none"
    gen_refs = ev.normalized_reference_urls_from_params(gp)
    if not gen_refs:
        return [], "generation_params.reference_images_empty", "none"
    ref_rows = ev.load_reference_selection_rows(run_dir)
    analysis_json = _read_json(run_dir / "analysis.json")
    if not isinstance(analysis_json, dict):
        analysis_json = None

    slots = ev.build_eval_reference_slots_from_generation_params(
        generation_inputs=gen_refs,
        phase4_run_dir=run_dir,
        analysis_json=analysis_json,
        reference_selection_rows=ref_rows,
    )
    locals_out = _paths_from_eval_slots(slots)
    if locals_out:
        return locals_out, None, "generation_params_rebuild"

    seen: set[str] = set(locals_out)
    ref_dir = run_dir / "reference_images"
    if ref_dir.is_dir():
        for p in sorted(ref_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                abs_p = str(p.resolve())
                if abs_p not in seen:
                    seen.add(abs_p)
                    locals_out.append(abs_p)
    if not locals_out:
        return [], "no_local_reference_paths_resolved", "none"
    return locals_out, None, "reference_images_dir"


def _prompt_from_run(run_dir: Path, gp: Dict[str, Any]) -> str:
    p = str(gp.get("prompt") or "").strip()
    if p:
        return p
    rp = run_dir / "refined_prompt.txt"
    if rp.is_file():
        t = rp.read_text(encoding="utf-8").strip()
        if t:
            return t
    return ""


def _augmented_details(run_dir: Path, ev: Any, n_refs: int, generation_params: Dict[str, Any]) -> Dict[str, Any]:
    analysis = _read_json(run_dir / "analysis.json")
    expected = 0
    if isinstance(analysis, dict):
        try:
            expected = int(ev.compute_expected_reference_images_count(analysis))
        except Exception:
            expected = 0
    sel_rows = ev.load_reference_selection_rows(run_dir)
    used = _summary_used_ref_count(run_dir)
    # One slot per I2I reference in generation_params.json, same order; metadata from analysis+selection.
    eval_slots: List[Dict[str, Any]] = []
    gen_refs = ev.normalized_reference_urls_from_params(generation_params)
    if gen_refs:
        eval_slots = ev.build_eval_reference_slots_from_generation_params(
            generation_inputs=gen_refs,
            phase4_run_dir=run_dir,
            analysis_json=analysis if isinstance(analysis, dict) else None,
            reference_selection_rows=sel_rows,
        )
    tk_slots = ev.build_eval_text_knowledge_slots_from_analysis(
        analysis if isinstance(analysis, dict) else None
    )
    return {
        "needs_search": bool((analysis or {}).get("needs_search")) if isinstance(analysis, dict) else False,
        "expected_reference_images_count": expected,
        "selected_reference_images_count": len(sel_rows) if isinstance(sel_rows, list) else 0,
        "used_reference_images_count": used if used > 0 else n_refs,
        "eval_reference_slots": eval_slots,
        "eval_text_knowledge_slots": tk_slots,
        "has_text_knowledge_gaps_for_eval": len(tk_slots) > 0,
    }


def _empty_stats() -> Dict[str, int]:
    return {
        "shard_files_seen": 0,
        "passed_difficulty_filter": 0,
        "missing_run_dir": 0,
        "missing_generation_params": 0,
        "unresolved_refs": 0,
        "empty_prompt": 0,
        "skipped_no_eval_context": 0,
        "ref_paths_from_reference_eval_lookup": 0,
        "ref_paths_from_generation_params_rebuild": 0,
        "ref_paths_from_reference_images_dir": 0,
        "rows_with_text_references": 0,
    }


def _merge_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _read_nonempty_jsonl_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def _atomic_write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=".ff_atomic_",
        suffix=".jsonl",
        delete=False,
    ) as tf:
        tmp = Path(tf.name)
        for row in lines:
            tf.write(row + "\n")
    tmp.replace(path)


def _split_train_test(
    train_path: Path,
    *,
    test_count: int,
    test_path: Path,
    random_split: bool,
    seed: Optional[int],
) -> None:
    lines = _read_nonempty_jsonl_lines(train_path)
    if len(lines) < test_count:
        raise SystemExit(
            f"train/test split: need at least {test_count} nonempty lines in {train_path}, got {len(lines)}"
        )
    if random_split:
        rng = random.Random(seed)
        rng.shuffle(lines)
    test_lines = lines[:test_count]
    train_lines = lines[test_count:]
    _atomic_write_lines(test_path, test_lines)
    _atomic_write_lines(train_path, train_lines)
    print(
        f"[split] {len(test_lines)} rows -> {test_path}, {len(train_lines)} rows -> {train_path}",
        file=sys.stderr,
    )


def _row_has_text_references(row: Dict[str, Any]) -> bool:
    ag = row.get("augmented_generation_details")
    if not isinstance(ag, dict):
        return False
    slots = ag.get("eval_text_knowledge_slots")
    return isinstance(slots, list) and len(slots) > 0


def _count_text_ref_rows_in_jsonl(path: Path) -> Tuple[int, int]:
    """Return (rows_with_text_references, total_nonempty_rows)."""
    n_text = 0
    n_total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n_total += 1
            row = json.loads(s)
            if not isinstance(row, dict):
                raise TypeError(
                    f"expected JSON object in {path}, line near total {n_total!r}, got {type(row).__name__}"
                )
            if _row_has_text_references(row):
                n_text += 1
    return n_text, n_total


def _run_prune_jsonl(
    jsonl_paths: List[Path],
    *,
    image_dir: Path,
    prune_workers: int,
    max_images: Optional[int],
    drop_pixels_percentile: Optional[int],
) -> None:
    existing = [p for p in jsonl_paths if p.is_file()]
    if not existing:
        return
    script = Path(__file__).resolve().parent / "prune_jsonl_broken_reference_images.py"
    if not script.is_file():
        raise SystemExit(f"prune script not found: {script}")
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--jsonl",
        *[str(p) for p in existing],
        "--image-dir",
        str(image_dir),
        "--workers",
        str(prune_workers),
    ]
    if max_images is not None:
        cmd.extend(["--max-images", str(max_images)])
    if drop_pixels_percentile is not None:
        cmd.extend(["--drop-pixels-above-percentile", str(drop_pixels_percentile)])
    print(f"[prune] {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd)


def _build_row_from_shard(
    shard_path: Path,
    toolgen_root: Path,
    min_visual_difficulty: float,
    ev: Any,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    st = _empty_stats()
    st["shard_files_seen"] = 1

    rec = _load_shard_record(shard_path)
    if rec is None:
        return None, st

    score = _visual_difficulty(rec)
    if score is None or score < min_visual_difficulty:
        return None, st
    st["passed_difficulty_filter"] = 1

    evc = rec.get("evaluation_context")
    if not isinstance(evc, dict):
        st["skipped_no_eval_context"] = 1
        return None, st

    p4 = rec.get("phase4")
    if not isinstance(p4, dict):
        st["missing_run_dir"] = 1
        return None, st
    rd = p4.get("run_dir")
    if not isinstance(rd, str) or not rd.strip():
        st["missing_run_dir"] = 1
        return None, st
    run_dir = Path(rd.strip()).resolve()
    if not run_dir.is_dir():
        st["missing_run_dir"] = 1
        return None, st
    if not (run_dir / "generation_params.json").is_file():
        st["missing_generation_params"] = 1
        return None, st

    gp = ev.load_generation_params_json(run_dir)
    if not isinstance(gp, dict):
        st["missing_generation_params"] = 1
        return None, st

    local_refs, err, ref_src = _resolve_local_reference_paths(run_dir, ev)
    if err:
        st["unresolved_refs"] = 1
        return None, st
    if ref_src == "reference_eval_lookup":
        st["ref_paths_from_reference_eval_lookup"] = 1
    elif ref_src == "generation_params_rebuild":
        st["ref_paths_from_generation_params_rebuild"] = 1
    elif ref_src == "reference_images_dir":
        st["ref_paths_from_reference_images_dir"] = 1

    prompt = _prompt_from_run(run_dir, gp)
    if not prompt.strip():
        st["empty_prompt"] = 1
        return None, st

    checklist = evc.get("verification_checklist")
    if not isinstance(checklist, list):
        checklist = []
    rubric = _rubric_dict_only(evc.get("evaluation_rubric"))

    mri = rec.get("metadata_row_index")
    req_idx = int(mri) if isinstance(mri, int) else -1
    sid = str(rec.get("sample_id") or "").strip() or shard_path.stem

    row: Dict[str, Any] = {
        "prompt": prompt,
        "image": local_refs,
        "user_prompt": str(rec.get("user_prompt") or "").strip(),
        "verification_checklist": [str(x) for x in checklist if isinstance(x, str) and x.strip()],
        "evaluation_rubric": json.dumps(rubric, ensure_ascii=False),
        "trajectory_id": sid,
        "request_index": req_idx,
        "augmented_generation_details": _augmented_details(run_dir, ev, len(local_refs), gp),
        "metadata_dataset_path": rec.get("metadata_dataset_path"),
        "metadata_row_index": mri,
        "phase4_run_dir": str(run_dir),
        "difficulty_with_ideal_visual_references": score,
    }
    ag = row["augmented_generation_details"]
    if (
        isinstance(ag, dict)
        and isinstance(ag.get("eval_text_knowledge_slots"), list)
        and len(ag["eval_text_knowledge_slots"]) > 0
    ):
        st["rows_with_text_references"] = 1
    return row, st


def _shard_worker_task(task: Tuple[str, str, float]) -> Dict[str, Any]:
    """Picklable worker: (shard_path_str, toolgen_root_str, min_visual_difficulty)."""
    shard_path_str, toolgen_root_str, min_vis = task
    shard_path = Path(shard_path_str)
    toolgen_root = Path(toolgen_root_str)
    _ensure_toolgen_phase5_on_path(toolgen_root)
    import evaluate_searchbetter_hard_direct as ev  # type: ignore[import-not-found]

    row, st = _build_row_from_shard(shard_path, toolgen_root, min_vis, ev)
    return {"row": row, "stats": st}


def main() -> None:
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **kwargs):  # type: ignore[misc]
            return it

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--toolgen-root",
        type=Path,
        default=Path("/primus_xpfs_workspace_T04/haozhe/ToolGen"),
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Directory of per-row *.json shards (default: TOOLGEN_ROOT/phase2_prompt_generation/difficulty_label_shards)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path (parent created).")
    parser.add_argument("--min-visual-difficulty", type=float, default=3.5)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Emit at most this many rows (0 = no cap). With --workers>1, all shards are still "
        "processed for aggregate stats; only the first N successful rows are written.",
    )
    parser.add_argument(
        "--max-shard-files",
        type=int,
        default=0,
        help="Process only the first N shard files after sort (0 = all). Dev shortcut.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process pool size (0 = min(32, CPU count)). Use 1 for single-process.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bar.")
    parser.add_argument(
        "--test-count",
        type=int,
        default=32,
        help="After writing --output, move this many lines to a test set (0 = no split). Default: 32.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=None,
        help="Test JSONL path (default: <parent of --output>/test.jsonl).",
    )
    parser.add_argument(
        "--split-random",
        action="store_true",
        help="Shuffle before taking the test split; use with --split-seed.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="RNG seed when --split-random is set.")
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip the reference-image prune pass (PIL decodable + max-images + optional pixel filter).",
    )
    parser.add_argument(
        "--prune-image-dir",
        type=Path,
        default=None,
        help="Image base dir for relative paths in prune (default: <parent of --output>/images).",
    )
    parser.add_argument(
        "--prune-max-images",
        type=int,
        default=3,
        help="prune script --max-images (0 = no per-row image-count cap; default: 3).",
    )
    parser.add_argument(
        "--prune-drop-pixels-percentile",
        type=int,
        default=98,
        help="prune script --drop-pixels-above-percentile (0 = off; default: 98).",
    )
    parser.add_argument(
        "--prune-workers",
        type=int,
        default=0,
        help="prune process pool size (0 = min(32, CPUs)).",
    )
    args = parser.parse_args()

    toolgen_root = args.toolgen_root.resolve()
    shard_dir = (
        args.shard_dir.resolve()
        if args.shard_dir is not None
        else (toolgen_root / "phase2_prompt_generation" / "difficulty_label_shards").resolve()
    )
    if not shard_dir.is_dir():
        raise SystemExit(f"shard-dir is not a directory: {shard_dir}")

    paths = _list_shard_paths(shard_dir)
    if args.max_shard_files and args.max_shard_files > 0:
        paths = paths[: args.max_shard_files]

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cpu = os.cpu_count() or 8
    workers = args.workers if args.workers > 0 else min(32, n_cpu)
    cap = args.max_rows if args.max_rows and args.max_rows > 0 else None

    tasks = [(str(p.resolve()), str(toolgen_root), float(args.min_visual_difficulty)) for p in paths]
    stats_total = _empty_stats()
    written = 0

    if workers <= 1:
        _ensure_toolgen_phase5_on_path(toolgen_root)
        import evaluate_searchbetter_hard_direct as ev  # type: ignore[import-not-found]

        it = paths if args.no_progress else tqdm(paths, desc="Shards", unit="file")
        with out_path.open("w", encoding="utf-8") as out_f:
            for shard_path in it:
                row, st = _build_row_from_shard(shard_path, toolgen_root, args.min_visual_difficulty, ev)
                _merge_stats(stats_total, st)
                if row is None:
                    continue
                if cap is not None and written >= cap:
                    continue
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    else:
        chunksize = max(1, len(tasks) // (workers * 8)) if tasks else 1
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool, out_path.open("w", encoding="utf-8") as out_f:
            imap_it = pool.imap(_shard_worker_task, tasks, chunksize=chunksize)
            if not args.no_progress:
                imap_it = tqdm(imap_it, total=len(tasks), desc="Shards", unit="file")
            for pack in imap_it:
                _merge_stats(stats_total, pack["stats"])
                row = pack.get("row")
                if row is None:
                    continue
                if cap is not None and written >= cap:
                    continue
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    stats_total["written"] = written
    print(json.dumps(stats_total, indent=2))
    print(f"Wrote {written} rows to {out_path}")

    train_path = out_path
    test_path = (args.test_output or train_path.parent / "test.jsonl").resolve()
    n_test = int(args.test_count)
    if n_test > 0:
        _split_train_test(
            train_path,
            test_count=n_test,
            test_path=test_path,
            random_split=bool(args.split_random),
            seed=int(args.split_seed) if args.split_seed is not None else None,
        )

    if not args.no_prune:
        to_prune: List[Path] = [train_path]
        if n_test > 0 and test_path.is_file():
            to_prune.append(test_path)
        img_d = args.prune_image_dir
        if img_d is None:
            img_d = train_path.parent / "images"
        else:
            img_d = Path(img_d).resolve()
        n_cpu2 = os.cpu_count() or 8
        pw = args.prune_workers if int(args.prune_workers) > 0 else min(32, n_cpu2)
        pm = int(args.prune_max_images)
        max_im: Optional[int] = None if pm <= 0 else pm
        pp = int(args.prune_drop_pixels_percentile)
        pct: Optional[int] = None if pp <= 0 else pp
        _run_prune_jsonl(
            to_prune,
            image_dir=img_d,
            prune_workers=pw,
            max_images=max_im,
            drop_pixels_percentile=pct,
        )

    t_tr, t_tot = _count_text_ref_rows_in_jsonl(train_path)
    print(
        f"Rows with text references (non-empty eval_text_knowledge_slots): "
        f"train {t_tr}/{t_tot} — {train_path}",
        file=sys.stderr,
    )
    if n_test > 0 and test_path.is_file():
        s_tr, s_tot = _count_text_ref_rows_in_jsonl(test_path)
        print(
            f"Rows with text references: test {s_tr}/{s_tot} — {test_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
