#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Build Flow-Factory Klein i2i / mixed T2I+I2I JSONL for NFT + ``toolgen_searchbetter_judge*`` rewards.

Plan
----
1. **Source** (pick one):
   - **Labeled JSONL** (default): ``phase2_prompt_generation/AA_synth_all_prompts_metadata_eval.difficulty_labeled.jsonl``
     — one object per line, same keys as legacy shards (``user_prompt``, ``evaluation_context``,
     ``phase4.run_dir``, ``difficulty_label``, …).
   - **Legacy shards**: ``difficulty_label_shards/*.json`` via ``--shard-dir``.

2. **Filter**: keep rows with ``difficulty_with_ideal_visual_references >= --min-visual-difficulty``

3. **Phase4 run**: require ``generation_params.json`` under ``run_dir``.

4. **Mixed T2I + I2I** (``--mix-t2i-i2i``, default on): if ``reference_images`` is empty, emit a **T2I**
   row (``image``: ``[]``, ``task_mode``: ``t2i``). Otherwise resolve conditioning refs like before
   (``task_mode``: ``i2i``).

5. **Prompts**: ``prompt`` = **refined** text for the generator (``refined_prompt.txt`` →
   ``refined_prompt.json`` → ``generation_params.prompt``). ``user_prompt`` = **original** task
   from the labeled record (for the judge).

6. **Output schema** — ``prompt``, ``image`` (abs paths list, possibly empty), ``user_prompt``,
   ``task_mode``, checklist, ``evaluation_rubric`` JSON string, ``trajectory_id``, ``request_index``,
   ``augmented_generation_details`` (``eval_reference_slots``, ``eval_text_knowledge_slots``, …).

Post steps: normalize (I2I rows only touch images), prune with ``--allow-empty-images`` when mixing,
train/test split.

Usage
-----
  PYTHONPATH=src python scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py \\
    --output dataset/klein_i2i_mix_visual_ge_35_v4/train.jsonl

  # Legacy shards:
  PYTHONPATH=src python scripts/build_klein_i2i_rl_dataset_from_difficulty_shards.py \\
    --shard-dir .../difficulty_label_shards \\
    --output dataset/legacy/train.jsonl

Or: ``bash scripts/build_klein_i2i_rl_dataset.sh``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    *,
    mix_allow_empty: bool,
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
        if mix_allow_empty:
            return [], None, "none"
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
        if mix_allow_empty:
            return [], None, "none"
        return [], "no_local_reference_paths_resolved", "none"
    return locals_out, None, "reference_images_dir"


def _refined_prompt_from_run(run_dir: Path, gp: Dict[str, Any]) -> str:
    """Generator conditioning: refined prompt files first, then generation_params.prompt."""
    rp_txt = run_dir / "refined_prompt.txt"
    if rp_txt.is_file():
        t = rp_txt.read_text(encoding="utf-8").strip()
        if t:
            return t
    rp_json = run_dir / "refined_prompt.json"
    raw_j = _read_json(rp_json)
    if isinstance(raw_j, dict):
        v = raw_j.get("refined_prompt")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(gp.get("prompt") or "").strip()


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
        "jsonl_lines_seen": 0,
        "passed_difficulty_filter": 0,
        "missing_run_dir": 0,
        "missing_generation_params": 0,
        "unresolved_refs": 0,
        "empty_prompt": 0,
        "skipped_no_eval_context": 0,
        "skipped_bad_jsonl": 0,
        "ref_paths_from_reference_eval_lookup": 0,
        "ref_paths_from_generation_params_rebuild": 0,
        "ref_paths_from_reference_images_dir": 0,
        "rows_with_text_references": 0,
        "rows_t2i": 0,
        "rows_i2i": 0,
    }


def _merge_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _row_image_paths(row: Dict[str, Any]) -> List[str]:
    """Absolute paths from row['image'] (str or list[str])."""
    raw = row.get("image")
    if raw is None:
        return []
    if isinstance(raw, str) and raw.strip():
        return [str(Path(raw.strip()).resolve())]
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(str(Path(x.strip()).resolve()))
        return out
    raise TypeError(
        f"row trajectory_id={row.get('trajectory_id')!r}: 'image' must be str, list[str], or null, "
        f"got {type(raw).__name__}"
    )


def _patch_eval_reference_slots_paths(
    row: Dict[str, Any],
    src_to_dst: Dict[str, str],
    failures: Dict[str, str],
) -> bool:
    """Rewrite local paths in ``eval_reference_slots`` to normalized PNGs. False => drop row."""
    ag = row.get("augmented_generation_details")
    if not isinstance(ag, dict):
        return True
    slots = ag.get("eval_reference_slots")
    if not isinstance(slots, list):
        return True
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        for key in ("reference_local_path", "generation_input"):
            v = slot.get(key)
            if not isinstance(v, str) or not v.strip():
                continue
            t = v.strip()
            low = t.lower()
            if low.startswith(("http://", "https://", "data:")):
                continue
            p = Path(t)
            if not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in failures:
                return False
            if rp in src_to_dst:
                slot[key] = src_to_dst[rp]
    return True


def _normalize_one_image_task(
    args: Tuple[str, str, int],
) -> Tuple[str, str, Optional[str], Optional[Tuple[int, int, int, int]]]:
    """Return (src_abs, dst_abs, err, meta). meta is (orig_w, orig_h, scaled_w, scaled_h) on success."""
    src_abs, dst_abs, side = args
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        try:
            _resample = Image.Resampling.LANCZOS  # Pillow >= 9.1
        except AttributeError:
            _resample = Image.LANCZOS  # type: ignore[attr-defined]
        def _as_rgb_on_white(src: Image.Image) -> Image.Image:
            """Palette/RGBA/LA + alpha to RGB flattened on white; avoids PIL palette+transparency warnings."""
            mode = src.mode
            if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in src.info):
                rgba = src.convert("RGBA")
                out = Image.new("RGB", rgba.size, (255, 255, 255))
                out.paste(rgba, mask=rgba.split()[3])
                return out
            return src.convert("RGB")

        with Image.open(src_abs) as im:
            im.load()
            im = _as_rgb_on_white(im)
            w, h = im.size
            if w <= 0 or h <= 0:
                return src_abs, dst_abs, "non_positive_size", None
            scale = min(float(side) / float(w), float(side) / float(h))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            resized = im.resize((nw, nh), _resample)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        ox = (side - nw) // 2
        oy = (side - nh) // 2
        canvas.paste(resized, (ox, oy))
        Path(dst_abs).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst_abs, format="PNG", optimize=True)
    except Exception as e:
        return src_abs, dst_abs, f"{type(e).__name__}: {e}", None
    return src_abs, dst_abs, None, (w, h, nw, nh)


def _normalize_jsonl_condition_images(
    jsonl_path: Path,
    dest_dir: Path,
    *,
    side: int,
    workers: int,
) -> None:
    """
    Rewrite ``jsonl_path`` so every ``image`` path points to a ``side`` x ``side`` RGB PNG
    (aspect-preserving scale, centered on white) under ``dest_dir``. Rows that reference a
    missing or failed source image are dropped.
    """
    try:
        from tqdm import tqdm as _tqdm_norm
    except ImportError:

        def _tqdm_norm(it=None, **kwargs):  # type: ignore[misc]
            return it if it is not None else []

    lines = _read_nonempty_jsonl_lines(jsonl_path)
    if not lines:
        return
    rows: List[Dict[str, Any]] = []
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"expected JSON object in {jsonl_path}, got {type(row).__name__}")
        rows.append(row)

    unique_src: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for p in _row_image_paths(row):
            if p not in seen:
                seen.add(p)
                unique_src.append(p)

    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, int]] = []
    for src in unique_src:
        h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        stem = Path(src).stem
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
        dst = dest_dir / f"{h}_{safe_stem}.png"
        tasks.append((src, str(dst), int(side)))

    src_to_dst: Dict[str, str] = {}
    failures: Dict[str, str] = {}
    padding_examples: List[Tuple[str, str, int, int, int, int]] = []
    n_cpu = os.cpu_count() or 8
    w = workers if workers > 0 else min(32, n_cpu)

    def _record_padding_example(s: str, d: str, meta: Tuple[int, int, int, int]) -> None:
        ow, oh, nw, nh = meta
        if len(padding_examples) >= 2:
            return
        if nw < side or nh < side:
            padding_examples.append((s, d, ow, oh, nw, nh))

    if w <= 1 or len(tasks) <= 1:
        for t in _tqdm_norm(tasks, desc="Normalize condition images", unit="img"):
            s, d, err, meta = _normalize_one_image_task(t)
            if err:
                failures[s] = err
            else:
                src_to_dst[s] = d
                if meta is not None:
                    _record_padding_example(s, d, meta)
    else:
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(_normalize_one_image_task, t) for t in tasks]
            for fut in _tqdm_norm(as_completed(futs), total=len(futs), desc="Normalize condition images", unit="img"):
                s, d, err, meta = fut.result()
                if err:
                    failures[s] = err
                else:
                    src_to_dst[s] = d
                    if meta is not None:
                        _record_padding_example(s, d, meta)

    if padding_examples:
        print(
            f"[normalize] White-padding examples (source -> {side}×{side} PNG, orig -> scaled on canvas):",
            file=sys.stderr,
        )
        for s, d, ow, oh, nw, nh in padding_examples:
            print(f"  {s}", file=sys.stderr)
            print(f"    -> {d}", file=sys.stderr)
            print(f"    ({ow}×{oh} -> content {nw}×{nh} on {side}×{side} white)", file=sys.stderr)
    else:
        print(
            f"[normalize] No white-padding examples to show (no sources needed letterboxing into {side}×{side}).",
            file=sys.stderr,
        )

    if failures:
        print(
            f"[normalize] {len(failures)}/{len(tasks)} source images failed (rows referencing them will be dropped)",
            file=sys.stderr,
        )
        for i, (s, e) in enumerate(sorted(failures.items())[:20]):
            print(f"  - {s}: {e}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)

    kept_lines: List[str] = []
    dropped = 0
    for row in rows:
        try:
            srcs = _row_image_paths(row)
        except TypeError:
            dropped += 1
            continue
        if not srcs:
            if not _patch_eval_reference_slots_paths(row, src_to_dst, failures):
                dropped += 1
                continue
            kept_lines.append(json.dumps(row, ensure_ascii=False))
            continue
        if any(s in failures for s in srcs):
            dropped += 1
            continue
        if any(s not in src_to_dst for s in srcs):
            dropped += 1
            continue
        new_paths = [src_to_dst[s] for s in srcs]
        raw = row.get("image")
        if isinstance(raw, str):
            row["image"] = new_paths[0] if len(new_paths) == 1 else new_paths
        else:
            row["image"] = new_paths
        if not _patch_eval_reference_slots_paths(row, src_to_dst, failures):
            dropped += 1
            continue
        kept_lines.append(json.dumps(row, ensure_ascii=False))

    _atomic_write_lines(jsonl_path, kept_lines)
    print(
        f"[normalize] wrote {len(src_to_dst)} files under {dest_dir}; "
        f"jsonl {jsonl_path}: {len(kept_lines)} rows kept, {dropped} dropped",
        file=sys.stderr,
    )


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
    allow_empty_images: bool,
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
    if allow_empty_images:
        cmd.append("--allow-empty-images")
    if max_images is not None:
        cmd.extend(["--max-images", str(max_images)])
    if drop_pixels_percentile is not None:
        cmd.extend(["--drop-pixels-above-percentile", str(drop_pixels_percentile)])
    print(f"[prune] {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd)


def _build_row_from_labeled_record(
    rec: Dict[str, Any],
    *,
    default_trajectory_stem: str,
    min_visual_difficulty: float,
    ev: Any,
    mix_t2i_i2i: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    st = _empty_stats()

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

    local_refs, err, ref_src = _resolve_local_reference_paths(run_dir, ev, mix_allow_empty=mix_t2i_i2i)
    if err:
        st["unresolved_refs"] = 1
        return None, st
    if ref_src == "reference_eval_lookup":
        st["ref_paths_from_reference_eval_lookup"] = 1
    elif ref_src == "generation_params_rebuild":
        st["ref_paths_from_generation_params_rebuild"] = 1
    elif ref_src == "reference_images_dir":
        st["ref_paths_from_reference_images_dir"] = 1

    prompt = _refined_prompt_from_run(run_dir, gp)
    if not prompt.strip():
        st["empty_prompt"] = 1
        return None, st

    if local_refs:
        st["rows_i2i"] = 1
    else:
        st["rows_t2i"] = 1

    checklist = evc.get("verification_checklist")
    if not isinstance(checklist, list):
        checklist = []
    rubric = _rubric_dict_only(evc.get("evaluation_rubric"))

    mri = rec.get("metadata_row_index")
    req_idx = int(mri) if isinstance(mri, int) else -1
    sid = str(rec.get("sample_id") or "").strip() or default_trajectory_stem

    task_mode = "i2i" if local_refs else "t2i"
    row: Dict[str, Any] = {
        "prompt": prompt,
        "image": local_refs,
        "user_prompt": str(rec.get("user_prompt") or "").strip(),
        "task_mode": task_mode,
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


def _build_row_from_shard(
    shard_path: Path,
    toolgen_root: Path,
    min_visual_difficulty: float,
    ev: Any,
    *,
    mix_t2i_i2i: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    st_head = _empty_stats()
    st_head["shard_files_seen"] = 1
    rec = _load_shard_record(shard_path)
    if rec is None:
        return None, st_head
    row, st = _build_row_from_labeled_record(
        rec,
        default_trajectory_stem=shard_path.stem,
        min_visual_difficulty=min_visual_difficulty,
        ev=ev,
        mix_t2i_i2i=mix_t2i_i2i,
    )
    st["shard_files_seen"] = 1
    return row, st


def _shard_worker_task(task: Tuple[str, str, float, int]) -> Dict[str, Any]:
    """Picklable worker: (shard_path_str, toolgen_root_str, min_visual_difficulty, mix_as_int)."""
    shard_path_str, toolgen_root_str, min_vis, mix_i = task
    shard_path = Path(shard_path_str)
    toolgen_root = Path(toolgen_root_str)
    _ensure_toolgen_phase5_on_path(toolgen_root)
    import evaluate_searchbetter_hard_direct as ev  # type: ignore[import-not-found]

    row, st = _build_row_from_shard(
        shard_path, toolgen_root, min_vis, ev, mix_t2i_i2i=bool(mix_i)
    )
    return {"row": row, "stats": st}


def _jsonl_worker_task(task: Tuple[str, str, float, int]) -> Dict[str, Any]:
    """Picklable worker: (line_str, toolgen_root_str, min_visual_difficulty, mix_as_int)."""
    line_str, toolgen_root_str, min_vis, mix_i = task
    toolgen_root = Path(toolgen_root_str)
    st = _empty_stats()
    st["jsonl_lines_seen"] = 1
    try:
        rec = json.loads(line_str)
    except json.JSONDecodeError:
        st["skipped_bad_jsonl"] = 1
        return {"row": None, "stats": st}
    if not isinstance(rec, dict):
        st["skipped_bad_jsonl"] = 1
        return {"row": None, "stats": st}
    _ensure_toolgen_phase5_on_path(toolgen_root)
    import evaluate_searchbetter_hard_direct as ev  # type: ignore[import-not-found]

    stem = str(rec.get("sample_id") or "").strip() or hashlib.sha256(line_str.encode()).hexdigest()[:16]
    row, st2 = _build_row_from_labeled_record(
        rec,
        default_trajectory_stem=stem,
        min_visual_difficulty=float(min_vis),
        ev=ev,
        mix_t2i_i2i=bool(mix_i),
    )
    for k, v in st2.items():
        st[k] = st.get(k, 0) + v
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
        help="Use per-row *.json shards under this directory (exclusive with default labeled-jsonl source).",
    )
    parser.add_argument(
        "--labeled-jsonl",
        type=Path,
        default=None,
        help=(
            "ToolGen difficulty-labeled JSONL (one object per line). "
            "Default when --shard-dir omitted: "
            "TOOLGEN_ROOT/phase2_prompt_generation/AA_synth_all_prompts_metadata_eval.difficulty_labeled.jsonl"
        ),
    )
    parser.add_argument(
        "--max-jsonl-lines",
        type=int,
        default=0,
        help="Process only the first N nonempty JSONL lines (0 = all). Dev shortcut.",
    )
    parser.set_defaults(mix_t2i_i2i=True)
    parser.add_argument(
        "--no-mix-t2i-i2i",
        action="store_false",
        dest="mix_t2i_i2i",
        help="Require reference images for every row (drop text-to-image runs).",
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
        default=64,
        help="After writing --output, move this many lines to a test set (0 = no split). Default: 64.",
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
    parser.add_argument(
        "--normalize-condition-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize/pad each condition image to a square white canvas (--normalize-size), rewrite JSONL paths "
        "(default: true). Use --no-normalize-condition-images to skip.",
    )
    parser.add_argument(
        "--normalize-size",
        type=int,
        default=512,
        help="Output width/height for normalized condition images (default: 512).",
    )
    parser.add_argument(
        "--normalize-subdir",
        type=str,
        default="condition_images_512",
        help="Subdir under the parent of --output for PNGs (default: condition_images_512). Ignored if "
        "--normalize-dest-dir is set.",
    )
    parser.add_argument(
        "--normalize-dest-dir",
        type=Path,
        default=None,
        help="Absolute output directory for normalized PNGs (default: <output-parent>/<normalize-subdir>).",
    )
    parser.add_argument(
        "--normalize-workers",
        type=int,
        default=0,
        help="Parallel workers for normalization (0 = min(32, CPUs); 1 = serial).",
    )
    args = parser.parse_args()

    toolgen_root = args.toolgen_root.resolve()
    mix_flag = bool(args.mix_t2i_i2i)
    mix_i = 1 if mix_flag else 0
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cpu = os.cpu_count() or 8
    workers = args.workers if args.workers > 0 else min(32, n_cpu)
    cap = args.max_rows if args.max_rows and args.max_rows > 0 else None
    stats_total = _empty_stats()
    written = 0

    if args.shard_dir is not None:
        shard_dir = args.shard_dir.resolve()
        if not shard_dir.is_dir():
            raise SystemExit(f"shard-dir is not a directory: {shard_dir}")
        paths = _list_shard_paths(shard_dir)
        if args.max_shard_files and args.max_shard_files > 0:
            paths = paths[: args.max_shard_files]

        tasks = [
            (str(p.resolve()), str(toolgen_root), float(args.min_visual_difficulty), mix_i) for p in paths
        ]
        if workers <= 1:
            _ensure_toolgen_phase5_on_path(toolgen_root)
            import evaluate_searchbetter_hard_direct as ev  # type: ignore[import-not-found]

            it = paths if args.no_progress else tqdm(paths, desc="Shards", unit="file")
            with out_path.open("w", encoding="utf-8") as out_f:
                for shard_path in it:
                    row, st = _build_row_from_shard(
                        shard_path, toolgen_root, args.min_visual_difficulty, ev, mix_t2i_i2i=mix_flag
                    )
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
    else:
        labeled_path = (
            args.labeled_jsonl.resolve()
            if args.labeled_jsonl is not None
            else (
                toolgen_root / "phase2_prompt_generation" / "AA_synth_all_prompts_metadata_eval.difficulty_labeled.jsonl"
            ).resolve()
        )
        if not labeled_path.is_file():
            raise SystemExit(f"labeled-jsonl not found: {labeled_path}\nPass --shard-dir for shard mode.")
        raw_lines: List[str] = []
        with labeled_path.open(encoding="utf-8") as lf:
            for line in lf:
                s = line.strip()
                if s:
                    raw_lines.append(s)
        if args.max_jsonl_lines and args.max_jsonl_lines > 0:
            raw_lines = raw_lines[: int(args.max_jsonl_lines)]

        tasks = [
            (ln, str(toolgen_root), float(args.min_visual_difficulty), mix_i) for ln in raw_lines
        ]
        if workers <= 1:
            it = raw_lines if args.no_progress else tqdm(raw_lines, desc="JSONL", unit="line")
            with out_path.open("w", encoding="utf-8") as out_f:
                for line_str in it:
                    pack = _jsonl_worker_task(
                        (line_str, str(toolgen_root), float(args.min_visual_difficulty), mix_i)
                    )
                    _merge_stats(stats_total, pack["stats"])
                    row = pack.get("row")
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
                imap_it = pool.imap(_jsonl_worker_task, tasks, chunksize=chunksize)
                if not args.no_progress:
                    imap_it = tqdm(imap_it, total=len(tasks), desc="JSONL", unit="line")
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

    # 1) Normalize condition images into dataset folder; rewrite JSONL to new paths; drop bad rows.
    if bool(args.normalize_condition_images):
        nd = args.normalize_dest_dir
        if nd is None:
            nd = train_path.parent / str(args.normalize_subdir)
        else:
            nd = Path(nd).resolve()
        _normalize_jsonl_condition_images(
            train_path,
            nd,
            side=int(args.normalize_size),
            workers=int(args.normalize_workers),
        )

    # 2) Prune broken / oversize references on the combined file *before* train/test split so test size is exact.
    if not args.no_prune:
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
            [train_path],
            image_dir=img_d,
            prune_workers=pw,
            max_images=max_im,
            drop_pixels_percentile=pct,
            allow_empty_images=mix_flag,
        )

    if n_test > 0:
        n_kept = len(_read_nonempty_jsonl_lines(train_path))
        if n_kept < n_test:
            raise SystemExit(
                f"train/test split: need at least {n_test} rows after normalize+prune "
                f"(for test set size), got {n_kept} — {train_path}"
            )
        _split_train_test(
            train_path,
            test_count=n_test,
            test_path=test_path,
            random_split=bool(args.split_random),
            seed=int(args.split_seed) if args.split_seed is not None else None,
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
