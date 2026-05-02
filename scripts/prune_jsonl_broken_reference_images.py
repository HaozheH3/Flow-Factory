#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Prune Flow-Factory JSONL rows whose reference ``image`` paths cannot be decoded
like training (PIL ``open`` + ``load`` + ``RGB``), matching
``GeneralDataset._preprocess_batch``.

Typical failure: ``PIL.UnidentifiedImageError`` on corrupt or truncated files
under ToolGen ``reference_images/``. Very large but valid references can hit
``PIL.Image.DecompressionBombError`` unless the worker relaxes
``Image.MAX_IMAGE_PIXELS`` (this script disables that cap for the decode check).

Steps
-----
1. Scan ``train.jsonl`` and ``test.jsonl`` under ``--dataset-dir`` (or explicit
   ``--jsonl`` files).
2. Collect **unique** resolved paths (``--image-dir`` only affects relative paths).
3. Check each path once in parallel (``--workers``); decodable images record ``width x height`` (PIL size).
4. Optional: log all decodable paths and print pixel-count distribution; optional drop of rows
   whose paths exceed a pixel-count percentile (``--drop-pixels-above-percentile``).
5. Optionally **unlink** on-disk files that exist but fail decode (``--delete-bad-files``).
6. Rewrite JSONL without pruned rows (atomic replace); report max image shape in kept data.

Examples
--------
  # Klein i2i dataset (same layout as train_search log)
  PYTHONPATH=src python scripts/prune_jsonl_broken_reference_images.py \\
    --dataset-dir dataset/klein_i2i_visual_ge_35

  Dry-run only:
  PYTHONPATH=src python scripts/prune_jsonl_broken_reference_images.py \\
    --dataset-dir dataset/klein_i2i_visual_ge_35 --dry-run

  Backup before rewrite:
  PYTHONPATH=src python scripts/prune_jsonl_broken_reference_images.py \\
    --dataset-dir dataset/klein_i2i_visual_ge_35 --backup-suffix .bak
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _resolve_path(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _image_paths_for_row(row: Dict[str, Any], image_dir: str) -> List[str]:
    raw = row.get("image")
    if raw is None:
        return []
    if isinstance(raw, str) and raw.strip():
        paths = [raw.strip()]
    elif isinstance(raw, list):
        paths = [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]
    else:
        raise TypeError(
            f"row trajectory_id={row.get('trajectory_id')!r}: 'image' must be str, list[str], or null, "
            f"got {type(raw).__name__}"
        )
    return [_resolve_path(image_dir, p) for p in paths]


def _check_image_worker(path: str) -> Tuple[str, Optional[str], Optional[Tuple[int, int]]]:
    """Return (path, err, (width, height)) on success, err and None for shape on failure."""
    if not os.path.isfile(path):
        return path, "missing_file", None
    from PIL import Image, UnidentifiedImageError

    # PIL's default cap (~179M pixels) rejects oversized but legitimate references
    # (e.g. wide composites). This offline script only verifies decodability on
    # trusted local paths, not arbitrary URLs.
    Image.MAX_IMAGE_PIXELS = None

    try:
        with Image.open(path) as im:
            im.load()
            im.convert("RGB")
            wh: Tuple[int, int] = (im.size[0], im.size[1])
    except UnidentifiedImageError as e:
        return path, f"UnidentifiedImageError: {e}", None
    except OSError as e:
        return path, f"OSError: {e}", None
    except ValueError as e:
        return path, f"ValueError: {e}", None
    except Image.DecompressionBombError as e:
        return path, f"DecompressionBombError: {e}", None
    return path, None, wh


def _percentile_linear(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 100], ``sorted_vals`` non-decreasing."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empty sorted_vals")
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))
    t = k - lo
    return float(sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t)


def _fmt_pixel_stats(pixels: Sequence[int]) -> str:
    s = sorted(pixels)
    pcts = (50, 90, 95, 98, 99, 100)
    parts = [f"count={len(pixels)}", f"min={s[0]}", f"max={s[-1]}"]
    if len(pixels) >= 2:
        parts.append(f"mean={statistics.fmean(s):.2f}")
        parts.append(f"pstdev={statistics.pstdev(s):.2f}")
    else:
        parts.append("mean=N/A pstdev=N/A")
    for pc in pcts:
        val = int(round(_percentile_linear(s, float(pc)))) if s else 0
        parts.append(f"p{pc}={val}")
    return "Pixel count distribution (width*height, decodable unique paths): " + ", ".join(parts)


def _iter_jsonl_objects(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"{path}: line {line_no}: expected JSON object, got {type(obj).__name__}")
            yield line_no, obj


def _collect_unique_paths(
    jsonl_files: Sequence[Path],
    image_dir: str,
) -> Tuple[Set[str], List[Tuple[Path, int, Dict[str, Any]]]]:
    unique: Set[str] = set()
    all_rows: List[Tuple[Path, int, Dict[str, Any]]] = []
    for jp in jsonl_files:
        for line_no, row in _iter_jsonl_objects(jp):
            all_rows.append((jp, line_no, row))
            for p in _image_paths_for_row(row, image_dir):
                unique.add(p)
    return unique, all_rows


def _verify_paths_parallel(
    paths: Sequence[str],
    workers: int,
) -> Dict[str, Tuple[Optional[str], Optional[Tuple[int, int]]]]:
    """Map path -> (error or None, (w,h) on success)."""
    workers = max(1, int(workers))
    if not paths:
        return {}
    chunksize = max(1, len(paths) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        triples = ex.map(_check_image_worker, paths, chunksize=chunksize)
    out: Dict[str, Tuple[Optional[str], Optional[Tuple[int, int]]]] = {}
    for path, err, wh in triples:
        if err is not None:
            out[path] = (err, None)
        else:
            if wh is None:
                raise RuntimeError(
                    f"internal: decodable path {path!r} returned no shape; expected (width, height)"
                )
            out[path] = (None, wh)
    return out


def _row_has_only_good_paths(
    row: Dict[str, Any],
    image_dir: str,
    bad_paths: Set[str],
    *,
    require_images: bool,
) -> bool:
    paths = _image_paths_for_row(row, image_dir)
    if require_images and not paths:
        return False
    return all(p not in bad_paths for p in paths)


def _max_shape_in_rows(
    rows: Sequence[Dict[str, Any]],
    image_dir: str,
    path_wh: Dict[str, Tuple[int, int]],
) -> Optional[Tuple[Tuple[int, int], int]]:
    best_px = -1
    best_wh: Optional[Tuple[int, int]] = None
    for row in rows:
        for p in _image_paths_for_row(row, image_dir):
            wh = path_wh.get(p)
            if wh is None:
                continue
            w, h = wh
            px = w * h
            if px > best_px:
                best_px = px
                best_wh = wh
    if best_wh is None or best_px < 0:
        return None
    return (best_wh, best_px)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory containing train.jsonl / test.jsonl (default: none; use --jsonl or set this).",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit JSONL files to scan/prune (overrides default train/test under dataset-dir).",
    )
    p.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Base dir for relative image paths (default: {dataset-dir}/images).",
    )
    p.add_argument("--workers", type=int, default=0, help="Process pool size (0 = min(32, CPU count)).")
    p.add_argument(
        "--allow-empty-images",
        action="store_true",
        help="Keep rows with no 'image' paths (default: drop them; unusual for Klein i2i).",
    )
    p.add_argument("--dry-run", action="store_true", help="Report only; do not write JSONL or delete files.")
    p.add_argument(
        "--delete-bad-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unlink existing files that fail PIL decode (default: true). Missing files are not created.",
    )
    p.add_argument(
        "--backup-suffix",
        type=str,
        default="",
        help="If set (e.g. .bak), copy each JSONL to the same name + suffix before replace.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Drop rows whose 'image' field resolves to more than N paths (default: no limit).",
    )
    p.add_argument(
        "--drop-pixels-above-percentile",
        type=int,
        default=None,
        metavar="P",
        help=(
            "If set (e.g. 98), among decodable unique paths, compute the P-th percentile of "
            "width*height and drop any row that references a path with strictly more pixels (top "
            "roughly 100-P percent by size). Default: do not filter by size."
        ),
    )
    p.add_argument(
        "--log-good-image-shapes",
        type=Path,
        default=None,
        help=(
            "Write one JSONL line per decodable path: path, width, height, pixels. "
            "If --drop-pixels-above-percentile is set and this is omitted, defaults to "
            "{dataset_dir}/prune_decodable_image_shapes.jsonl"
        ),
    )
    args = p.parse_args()

    if args.jsonl:
        jsonl_files = [x.resolve() for x in args.jsonl]
        dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else jsonl_files[0].parent
    elif args.dataset_dir:
        dataset_dir = args.dataset_dir.resolve()
        jsonl_files = []
        for name in ("train.jsonl", "test.jsonl"):
            cand = dataset_dir / name
            if cand.is_file():
                jsonl_files.append(cand)
        if not jsonl_files:
            raise SystemExit(f"No train.jsonl or test.jsonl under {dataset_dir}")
    else:
        raise SystemExit("Provide --dataset-dir or one or more --jsonl paths.")

    image_dir = args.image_dir
    if image_dir is None:
        image_dir = str(dataset_dir / "images")
    else:
        image_dir = os.path.expanduser(image_dir)

    require_images = not args.allow_empty_images

    n_cpu = os.cpu_count() or 8
    workers = args.workers if args.workers > 0 else min(32, n_cpu)

    unique_paths, all_rows = _collect_unique_paths(jsonl_files, image_dir)
    path_results = _verify_paths_parallel(sorted(unique_paths), workers=workers)

    bad_paths: Set[str] = {path for path, (err, _wh) in path_results.items() if err is not None}
    bad_details: Dict[str, str] = {}
    for path in sorted(bad_paths):
        err, _ = path_results[path]
        if err is None:
            raise RuntimeError(f"internal: path in bad_paths but err is None: {path!r}")
        bad_details[path] = err

    path_wh: Dict[str, Tuple[int, int]] = {
        path: wh for path, (err, wh) in path_results.items() if err is None and wh is not None
    }
    for path, (err, wh) in path_results.items():
        if err is None and wh is None:
            raise RuntimeError(f"internal: decodable path {path!r} missing size tuple")

    print(f"Unique image paths checked: {len(unique_paths)}")
    print(f"Bad paths (missing or unloadable): {len(bad_paths)}")

    deletable: List[str] = []
    for path, (err, _wh) in path_results.items():
        if err is None:
            continue
        if err == "missing_file":
            continue
        if os.path.isfile(path):
            deletable.append(path)

    drop_pc = args.drop_pixels_above_percentile
    if drop_pc is not None and not 0 < drop_pc < 100:
        raise SystemExit(
            f"--drop-pixels-above-percentile must be between 1 and 99 inclusive, got {drop_pc!r}"
        )

    big_paths: Set[str] = set()
    if drop_pc is not None:
        if not path_wh:
            print(
                "[prune] No decodable image paths; skipping --drop-pixels-above-percentile "
                "(e.g. all rows are text-to-image with empty `image`).",
                file=sys.stderr,
            )
            drop_pc = None
    if drop_pc is not None:
        pixel_counts = [wh[0] * wh[1] for wh in path_wh.values()]
        srt_pc = sorted(pixel_counts)
        pixel_threshold = _percentile_linear(srt_pc, float(drop_pc))
        big_paths = {pt for pt, wh in path_wh.items() if float(wh[0] * wh[1]) > pixel_threshold + 1e-9}
        n_good_paths = len(path_wh)
        n_big_paths = len(big_paths)
        print(
            f"Pixel threshold at {drop_pc}th percentile: {pixel_threshold:.2f} pixels "
            f"({n_big_paths}/{n_good_paths} unique decodable paths strictly above threshold)"
        )
    if path_wh:
        pxs = [wh[0] * wh[1] for wh in path_wh.values()]
        print(_fmt_pixel_stats(pxs))
        srt = sorted(pxs)
        if len(srt) >= 10 and len(srt) <= 2000:
            dec = [_percentile_linear(srt, d * 10.0) for d in range(1, 11)]
            print(
                "Decile boundaries (pixel count, d10..d100): "
                + ", ".join(f"d{d*10}={int(round(v))}" for d, v in zip(range(1, 11), dec))
            )

    shape_log = args.log_good_image_shapes
    if shape_log is None and drop_pc is not None:
        shape_log = dataset_dir / "prune_decodable_image_shapes.jsonl"
    if shape_log is not None and path_wh:
        shape_log = shape_log.resolve()
        shape_log.parent.mkdir(parents=True, exist_ok=True)
        with shape_log.open("w", encoding="utf-8") as lf:
            for path in sorted(path_wh):
                w, h = path_wh[path]
                px = w * h
                rec: Dict[str, Any] = {
                    "path": path,
                    "width": w,
                    "height": h,
                    "pixels": px,
                }
                if drop_pc is not None:
                    rec["dropped_by_percentile"] = path in big_paths
                json.dump(rec, lf, ensure_ascii=False)
                lf.write("\n")
        print(f"Wrote {len(path_wh)} decodable path shape records to {shape_log}")

    rows_by_file: Dict[Path, List[Dict[str, Any]]] = {jp: [] for jp in jsonl_files}
    for jp, _line_no, row in all_rows:
        rows_by_file[jp].append(row)

    max_images = args.max_images
    dropped: Dict[str, int] = {str(jp): 0 for jp in jsonl_files}
    dropped_excess: Dict[str, int] = {str(jp): 0 for jp in jsonl_files}
    dropped_big: Dict[str, int] = {str(jp): 0 for jp in jsonl_files}
    kept: Dict[str, int] = {str(jp): 0 for jp in jsonl_files}

    def _exceeds_max_images(row: Dict[str, Any]) -> bool:
        if max_images is None:
            return False
        return len(_image_paths_for_row(row, image_dir)) > max_images

    def _row_is_kept(row: Dict[str, Any]) -> bool:
        if _exceeds_max_images(row):
            return False
        if not _row_has_only_good_paths(row, image_dir, bad_paths, require_images=require_images):
            return False
        for p in _image_paths_for_row(row, image_dir):
            if p in big_paths:
                return False
        return True

    for jp in jsonl_files:
        for row in rows_by_file[jp]:
            if _exceeds_max_images(row):
                dropped_excess[str(jp)] += 1
            elif not _row_has_only_good_paths(row, image_dir, bad_paths, require_images=require_images):
                dropped[str(jp)] += 1
            elif any(p in big_paths for p in _image_paths_for_row(row, image_dir)):
                dropped_big[str(jp)] += 1
            else:
                kept[str(jp)] += 1

    if max_images is not None:
        total_excess = sum(dropped_excess[str(jp)] for jp in jsonl_files)
        print(
            f"Rows dropped (more than {max_images} images): {total_excess} "
            f"({', '.join(f'{jp.name}: {dropped_excess[str(jp)]}' for jp in jsonl_files)})"
        )
    if drop_pc is not None:
        total_big = sum(dropped_big[str(jp)] for jp in jsonl_files)
        print(
            f"Rows dropped (image pixel count above {drop_pc}th percentile / large size): {total_big} "
            f"({', '.join(f'{jp.name}: {dropped_big[str(jp)]}' for jp in jsonl_files)})"
        )
    for path, reason in list(bad_details.items())[:50]:
        print(f"  BAD: {path}\n       {reason}")
    if len(bad_details) > 50:
        print(f"  ... ({len(bad_details) - 50} more)")

    if args.dry_run:
        print("Dry-run: no files deleted, JSONL not modified.")
        for jp in jsonl_files:
            de = dropped_excess[str(jp)]
            d = dropped[str(jp)]
            db = dropped_big[str(jp)]
            tot = de + d + db
            parts = [f"bad/empty: {d}"]
            if max_images is not None:
                parts.append(f"excess images: {de}")
            if drop_pc is not None:
                parts.append(f"large pixel: {db}")
            print(f"  {jp.name}: would keep {kept[str(jp)]}, drop {tot} (" + ", ".join(parts) + ")")
        all_would_keep = [r for jp in jsonl_files for r in rows_by_file[jp] if _row_is_kept(r)]
        mx = _max_shape_in_rows(all_would_keep, image_dir, path_wh)
        if mx is not None:
            (w, h), px = mx
            print(
                f"Max image shape in would-be-kept data: {w}x{h} width x height "
                f"({px} total pixels)."
            )
        else:
            print("Max image shape in would-be-kept data: N/A")
        return

    deleted = 0
    if args.delete_bad_files:
        for path in deletable:
            os.unlink(path)
            deleted += 1
        print(f"Deleted {deleted} corrupt on-disk file(s).")

    for jp in jsonl_files:
        out_rows = [row for row in rows_by_file[jp] if _row_is_kept(row)]
        if args.backup_suffix:
            bak = jp.with_name(jp.name + args.backup_suffix)
            shutil.copy2(jp, bak)
            print(f"Backed up {jp} -> {bak}")
        tmp = jp.with_suffix(jp.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(jp)
        de = dropped_excess[str(jp)]
        d = dropped[str(jp)]
        db = dropped_big[str(jp)]
        tot_dropped = de + d + db
        parts = [f"bad/empty: {d}"]
        if max_images is not None:
            parts.append(f"excess: {de}")
        if drop_pc is not None:
            parts.append(f"large pixel: {db}")
        print(
            f"Wrote {len(out_rows)} rows to {jp} (dropped {tot_dropped}: " + ", ".join(parts) + ")."
        )

    all_kept = [r for jp in jsonl_files for r in rows_by_file[jp] if _row_is_kept(r)]
    mx2 = _max_shape_in_rows(all_kept, image_dir, path_wh)
    if mx2 is not None:
        w, h = mx2[0]
        px2 = mx2[1]
        print(
            f"Max image shape in kept data (after this run): {w}x{h} width x height "
            f"({px2} total pixels)."
        )
    else:
        print("Max image shape in kept data (after this run): N/A")


if __name__ == "__main__":
    main()
