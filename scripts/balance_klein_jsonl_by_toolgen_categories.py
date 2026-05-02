#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Build **multi-label** balanced subsets of a Klein/ToolGen JSONL, with separate quotas for **I2I** and
**T2I** rows (by ``task_mode``).

For each modality, each **category code** in ``evaluation_context.primary_categories`` defines a list
of rows that include that tag; sampling **round-robins** across those lists and **deduplicates** by
``trajectory_id``, then fills shortfalls at random from the remaining rows of that modality only.

Defaults: **6000** I2I rows and **1000** T2I rows (7000 unique trajectories total).

Examples
--------
  python scripts/balance_klein_jsonl_by_toolgen_categories.py \\
    --input dataset/klein_i2i_mix_visual_ge_35_v4/train.jsonl \\
    --labeled-jsonl ../ToolGen/phase2_prompt_generation/AA_synth_all_prompts_metadata_eval.difficulty_labeled.jsonl \\
    --output-i2i dataset/klein_i2i_mix_visual_ge_35_v4/train_multilabel_i2i_6k.jsonl \\
    --output-t2i dataset/klein_i2i_mix_visual_ge_35_v4/train_multilabel_t2i_1k.jsonl \\
    --i2i-total 6000 --t2i-total 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set


def _labels_from_eval_context(
    evc: Any,
    *,
    include_secondary: bool,
) -> List[str]:
    if not isinstance(evc, dict):
        return []
    out: List[str] = []
    raw_pri = evc.get("primary_categories")
    if isinstance(raw_pri, list):
        for x in raw_pri:
            s = str(x).strip()
            if s:
                out.append(s)
    if include_secondary:
        raw_sec = evc.get("secondary_categories")
        if isinstance(raw_sec, list):
            for x in raw_sec:
                s = str(x).strip()
                if s:
                    out.append(s)
    return list(dict.fromkeys(out))


def _load_sample_id_to_labels(labeled_path: Path, *, include_secondary: bool) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with labeled_path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("sample_id") or "").strip()
            if not sid:
                continue
            evc = rec.get("evaluation_context")
            out[sid] = _labels_from_eval_context(evc, include_secondary=include_secondary)
    return out


def _multilabel_round_robin_sample(
    rows: List[Dict[str, Any]],
    sample_id_to_labels: Dict[str, List[str]],
    *,
    total: int,
    rng: random.Random,
    no_label_bucket: str = "__no_label__",
    missing_labeled_bucket: str = "__missing_labeled__",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tid_to_row: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tid = str(r.get("trajectory_id") or "").strip()
        if tid:
            tid_to_row[tid] = r

    label_to_tids: DefaultDict[str, List[str]] = defaultdict(list)
    missing_labeled = 0
    for tid, row in tid_to_row.items():
        labels = sample_id_to_labels.get(tid)
        if labels is None:
            missing_labeled += 1
            label_to_tids[missing_labeled_bucket].append(tid)
            continue
        if not labels:
            label_to_tids[no_label_bucket].append(tid)
            continue
        for lab in labels:
            label_to_tids[lab].append(tid)

    # Unique tids per label list (first occurrence wins order), then shuffle within label
    for lab in list(label_to_tids.keys()):
        label_to_tids[lab] = list(dict.fromkeys(label_to_tids[lab]))
        rng.shuffle(label_to_tids[lab])

    labels_order = sorted(label_to_tids.keys(), key=lambda x: (x.startswith("__"), x))
    ptr: Dict[str, int] = {lab: 0 for lab in labels_order}

    chosen_tids: List[str] = []
    chosen_set: Set[str] = set()

    def try_take_one_from(lab: str) -> bool:
        nonlocal chosen_tids, chosen_set
        lst = label_to_tids[lab]
        p = ptr[lab]
        while p < len(lst):
            tid = lst[p]
            p += 1
            ptr[lab] = p
            if tid not in chosen_set:
                chosen_set.add(tid)
                chosen_tids.append(tid)
                return True
        return False

    # Round-robin passes until total reached or no progress
    while len(chosen_tids) < total:
        progressed = False
        for lab in labels_order:
            if len(chosen_tids) >= total:
                break
            if try_take_one_from(lab):
                progressed = True
        if not progressed:
            break

    if len(chosen_tids) < total:
        pool = [tid for tid in tid_to_row if tid not in chosen_set]
        rng.shuffle(pool)
        for tid in pool:
            if len(chosen_tids) >= total:
                break
            chosen_set.add(tid)
            chosen_tids.append(tid)

    if len(chosen_tids) > total:
        chosen_tids = rng.sample(chosen_tids, total)
        chosen_set = set(chosen_tids)

    rng.shuffle(chosen_tids)
    out_rows = [tid_to_row[tid] for tid in chosen_tids]

    # Coverage: incidence of each label in the selected set (a row with k primary tags counts k times)
    label_incidence_final: Counter[str] = Counter()
    for tid in chosen_tids:
        labs = sample_id_to_labels.get(tid)
        if labs is None:
            label_incidence_final[missing_labeled_bucket] += 1
        elif not labs:
            label_incidence_final[no_label_bucket] += 1
        else:
            for lab in labs:
                label_incidence_final[lab] += 1

    meta = {
        "input_rows": len(rows),
        "distinct_trajectory_ids": len(tid_to_row),
        "multilabel_lists": len(labels_order),
        "rows_per_label_list_min": min((len(label_to_tids[k]) for k in labels_order), default=0),
        "rows_per_label_list_max": max((len(label_to_tids[k]) for k in labels_order), default=0),
        "missing_labeled_join_rows": missing_labeled,
        "output_rows": len(out_rows),
        "labels_order": labels_order,
        "label_list_sizes": {k: len(label_to_tids[k]) for k in labels_order},
        "label_incidence_in_output_top40": dict(label_incidence_final.most_common(40)),
    }
    return out_rows, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--labeled-jsonl", type=Path, required=True)
    ap.add_argument(
        "--output-i2i",
        type=Path,
        default=None,
        help="Write I2I-balanced JSONL here (required if --i2i-total > 0).",
    )
    ap.add_argument(
        "--output-t2i",
        type=Path,
        default=None,
        help="Write T2I-balanced JSONL here (required if --t2i-total > 0).",
    )
    ap.add_argument("--i2i-total", type=int, default=6000, help="Target I2I row count (0 = skip I2I output).")
    ap.add_argument("--t2i-total", type=int, default=1000, help="Target T2I row count (0 = skip T2I output).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-secondary-categories",
        action="store_true",
        help="Also treat evaluation_context.secondary_categories as labels (wider multi-label).",
    )
    ap.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Optional path to write combined sampling statistics JSON.",
    )
    args = ap.parse_args()

    i2i_n = int(args.i2i_total)
    t2i_n = int(args.t2i_total)
    if i2i_n < 0 or t2i_n < 0:
        raise SystemExit("--i2i-total and --t2i-total must be non-negative")
    if i2i_n == 0 and t2i_n == 0:
        raise SystemExit("at least one of --i2i-total or --t2i-total must be positive")
    if i2i_n > 0 and args.output_i2i is None:
        raise SystemExit("--output-i2i is required when --i2i-total > 0")
    if t2i_n > 0 and args.output_t2i is None:
        raise SystemExit("--output-t2i is required when --t2i-total > 0")

    rows: List[Dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if not isinstance(row, dict):
                raise TypeError("expected JSON object per line")
            rows.append(row)

    i2i_rows = [r for r in rows if str(r.get("task_mode") or "").strip() == "i2i"]
    t2i_rows = [r for r in rows if str(r.get("task_mode") or "").strip() == "t2i"]
    n_unknown = len(rows) - len(i2i_rows) - len(t2i_rows)

    sample_id_to_labels = _load_sample_id_to_labels(
        args.labeled_jsonl.resolve(),
        include_secondary=bool(args.include_secondary_categories),
    )

    combined: Dict[str, Any] = {
        "input_path": str(args.input.resolve()),
        "labeled_jsonl": str(args.labeled_jsonl.resolve()),
        "seed": int(args.seed),
        "include_secondary_categories": bool(args.include_secondary_categories),
        "input_rows_all_modality": len(rows),
        "input_rows_i2i": len(i2i_rows),
        "input_rows_t2i": len(t2i_rows),
        "input_rows_unknown_task_mode": n_unknown,
    }

    if i2i_n > 0:
        rng_i2i = random.Random(int(args.seed) ^ 0x33449056)
        out_i2i, meta_i2i = _multilabel_round_robin_sample(
            i2i_rows,
            sample_id_to_labels,
            total=i2i_n,
            rng=rng_i2i,
        )
        meta_i2i["task_mode"] = "i2i"
        meta_i2i["total_requested"] = i2i_n
        if len(out_i2i) < i2i_n:
            meta_i2i["warning"] = (
                f"only {len(out_i2i)} unique I2I rows available (requested {i2i_n})"
            )
        op = args.output_i2i.resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8") as wf:
            for row in out_i2i:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta_i2i["output_path"] = str(op)
        combined["i2i"] = meta_i2i

    if t2i_n > 0:
        rng_t2i = random.Random(int(args.seed) ^ 0x88995566)
        out_t2i, meta_t2i = _multilabel_round_robin_sample(
            t2i_rows,
            sample_id_to_labels,
            total=t2i_n,
            rng=rng_t2i,
        )
        meta_t2i["task_mode"] = "t2i"
        meta_t2i["total_requested"] = t2i_n
        if len(out_t2i) < t2i_n:
            meta_t2i["warning"] = (
                f"only {len(out_t2i)} unique T2I rows available (requested {t2i_n})"
            )
        op = args.output_t2i.resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8") as wf:
            for row in out_t2i:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta_t2i["output_path"] = str(op)
        combined["t2i"] = meta_t2i

    print(json.dumps(combined, indent=2, ensure_ascii=False))
    if args.meta_json is not None:
        mp = args.meta_json.resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
