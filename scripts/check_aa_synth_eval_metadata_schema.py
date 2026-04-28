#!/usr/bin/env python3
# Copyright 2026 Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Validate ``AA_synth_all_prompts_metadata_eval.jsonl`` (or compatible) for ToolGen-style judge metadata.

Contract for ``toolgen_searchbetter_judge*`` rewards (text side):
  - ``user_prompt`` (str)
  - ``verification_checklist`` (list[str])
  - ``evaluation_rubric`` (dict[str, dict] with weight/description)

This file nests checklist/rubric under ``evaluation_context``; map them at dataset merge time.

Sufficiency:
  - **Judge prompt text**: yes, after unwrapping ``evaluation_context`` (and optional trajectory fields).
  - **Full RL row**: not alone — you still need ``prompt``, ``image``, ``condition_images`` (and usually
    ``metadata_dataset_path``/index only if you call ``load_request_meta``; rewards can use flattened
    checklist/rubric from this file without that lookup).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _as_str_list(x: Any, *, field: str, line_no: int) -> List[str]:
    if not isinstance(x, list):
        raise ValueError(f"line {line_no}: {field} must be a list, got {type(x).__name__}")
    out: List[str] = []
    for i, item in enumerate(x):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"line {line_no}: {field}[{i}] must be a non-empty str, got {type(item).__name__}: {item!r}"
            )
        out.append(item.strip())
    return out


def _as_rubric(x: Any, *, field: str, line_no: int) -> Dict[str, Dict[str, Any]]:
    if not isinstance(x, dict):
        raise ValueError(f"line {line_no}: {field} must be a dict, got {type(x).__name__}")
    if not x:
        raise ValueError(f"line {line_no}: {field} must be non-empty")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in x.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"line {line_no}: {field} has invalid key {k!r}")
        if not isinstance(v, dict):
            raise ValueError(
                f"line {line_no}: {field}[{k!r}] must be a dict, got {type(v).__name__}"
            )
        out[k.strip()] = dict(v)
    return out


def validate_record(obj: Dict[str, Any], *, line_no: int, check_run_dirs: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    sid = obj.get("sample_id")
    if not isinstance(sid, str) or not sid.strip():
        errors.append(f"line {line_no}: missing or invalid sample_id")

    up = obj.get("user_prompt")
    if not isinstance(up, str) or not up.strip():
        errors.append(f"line {line_no}: missing or empty user_prompt")

    evc = obj.get("evaluation_context")
    if not isinstance(evc, dict):
        errors.append(f"line {line_no}: evaluation_context must be a dict")
        return errors, warns

    cl_raw = evc.get("verification_checklist")
    if cl_raw is None:
        errors.append(f"line {line_no}: evaluation_context.verification_checklist is required")
        cl = []
    else:
        try:
            cl = _as_str_list(cl_raw, field="verification_checklist", line_no=line_no)
        except ValueError as e:
            errors.append(str(e))
            cl = []
    if not cl and not errors:
        warns.append(f"line {line_no}: empty verification_checklist (judge allows it but weak signal)")

    rub: Dict[str, Dict[str, Any]] = {}
    raw_rub = evc.get("evaluation_rubric")
    if raw_rub is None:
        warns.append(f"line {line_no}: missing evaluation_rubric (empty dict is allowed for the reward)")
    elif not isinstance(raw_rub, dict):
        errors.append(f"line {line_no}: evaluation_rubric must be a dict, got {type(raw_rub).__name__}")
    elif not raw_rub:
        warns.append(f"line {line_no}: empty evaluation_rubric (reward accepts {{}})")
    else:
        bad_scalar_keys: List[str] = []
        for k, v in raw_rub.items():
            if not isinstance(k, str) or not k.strip():
                errors.append(f"line {line_no}: evaluation_rubric has invalid key {k!r}")
                continue
            if isinstance(v, dict):
                rub[k.strip()] = dict(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                bad_scalar_keys.append(f"{k.strip()}={v!r}")
            else:
                errors.append(
                    f"line {line_no}: evaluation_rubric[{k!r}] must be a dict or numeric weight, "
                    f"got {type(v).__name__}"
                )
        if bad_scalar_keys:
            warns.append(
                f"line {line_no}: evaluation_rubric has scalar weight entries (skipped for schema; "
                f"normalize_rubric in training also drops these): {', '.join(bad_scalar_keys[:6])}"
                + (" ..." if len(bad_scalar_keys) > 6 else "")
            )

    if rub:
        weights: List[float] = []
        for dim, meta in rub.items():
            w = meta.get("weight", 1.0)
            if isinstance(w, (int, float)):
                weights.append(float(w))
            else:
                try:
                    weights.append(float(w))
                except (TypeError, ValueError):
                    errors.append(f"line {line_no}: rubric[{dim!r}].weight not numeric: {w!r}")
        if weights and abs(sum(weights) - 1.0) > 0.05:
            warns.append(
                f"line {line_no}: rubric weights sum to {sum(weights):.4f} (expected ~1.0); "
                "judge still works but review weighting."
            )

    p4 = obj.get("phase4")
    if p4 is not None:
        if not isinstance(p4, dict):
            errors.append(f"line {line_no}: phase4 must be a dict or absent")
        else:
            rd = p4.get("run_dir")
            if rd is not None and not isinstance(rd, str):
                errors.append(f"line {line_no}: phase4.run_dir must be str or absent")
            elif check_run_dirs and isinstance(rd, str) and rd.strip():
                p = Path(rd.strip())
                if not p.is_dir():
                    warns.append(f"line {line_no}: phase4.run_dir not a directory: {rd}")

    mdp = obj.get("metadata_dataset_path")
    if mdp is not None and not isinstance(mdp, str):
        errors.append(f"line {line_no}: metadata_dataset_path must be str or absent")
    mri = obj.get("metadata_row_index")
    if mri is not None and not isinstance(mri, int):
        errors.append(f"line {line_no}: metadata_row_index must be int or absent")

    return errors, warns


def _smoke_flow_factory_prompt(path: Path) -> None:
    from flow_factory.rewards.toolgen_searchbetter_judge_common import (
        build_eval_prompt_text,
        build_visual_reference_context_minimal,
    )

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            break
        else:
            raise SystemExit("smoke-flow-prompt: empty jsonl")

    if not isinstance(obj, dict):
        raise SystemExit("smoke-flow-prompt: first row not an object")
    evc = obj.get("evaluation_context")
    if not isinstance(evc, dict):
        raise SystemExit("smoke-flow-prompt: missing evaluation_context")

    row = {
        "trajectory_id": str(obj.get("sample_id") or "unknown"),
        "user_prompt": str(obj.get("user_prompt") or ""),
        "request_index": obj.get("metadata_row_index") if isinstance(obj.get("metadata_row_index"), int) else -1,
    }
    checklist = evc.get("verification_checklist") or []
    rubric = evc.get("evaluation_rubric") or {}
    if not isinstance(checklist, list):
        raise SystemExit("smoke-flow-prompt: verification_checklist not a list")
    if not isinstance(rubric, dict):
        raise SystemExit("smoke-flow-prompt: evaluation_rubric not a dict")

    text = build_eval_prompt_text(
        row=row,
        verification_checklist=checklist,
        evaluation_rubric=rubric,
        visual_context=build_visual_reference_context_minimal(False),
        variant="training",
        reference_slot_urls=(),
    )
    print(f"smoke-flow-prompt: built {len(text)} chars from first row sample_id={row['trajectory_id']!r}")
    print(text[:400].rstrip() + "\n...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to AA_synth_all_prompts_metadata_eval.jsonl (or compatible)",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 = scan entire file")
    parser.add_argument(
        "--check-run-dirs",
        action="store_true",
        help="Warn if phase4.run_dir is set but does not exist on disk",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero if any warning is emitted",
    )
    parser.add_argument(
        "--smoke-flow-prompt",
        action="store_true",
        help="After validation, build judge instruction text for the first row via flow_factory (repo on PYTHONPATH)",
    )
    args = parser.parse_args()

    path = args.jsonl.resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    total = 0
    err_count = 0
    warn_count = 0
    all_errors: List[str] = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if args.max_rows > 0 and total >= args.max_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {line_no}: invalid JSON: {e}")
                err_count += 1
                total += 1
                continue
            if not isinstance(obj, dict):
                all_errors.append(f"line {line_no}: top-level value must be object, got {type(obj).__name__}")
                err_count += 1
                total += 1
                continue

            errs, warns = validate_record(obj, line_no=line_no, check_run_dirs=args.check_run_dirs)
            if errs:
                err_count += len(errs)
                all_errors.extend(errs)
            if warns:
                warn_count += len(warns)
                for w in warns:
                    print(f"WARN {w}", file=sys.stderr)
            total += 1

    print(f"Scanned rows: {total} in {path}")
    if all_errors:
        print(f"ERRORS ({len(all_errors)}):", file=sys.stderr)
        for e in all_errors[:200]:
            print(e, file=sys.stderr)
        if len(all_errors) > 200:
            print(f"... and {len(all_errors) - 200} more", file=sys.stderr)
        raise SystemExit(1)

    if args.fail_on_warn and warn_count > 0:
        raise SystemExit(f"fail-on-warn: {warn_count} warning(s)")

    print("Schema check: OK (no errors).")
    if warn_count:
        print(f"Warnings: {warn_count} (see stderr)")

    if args.smoke_flow_prompt:
        _smoke_flow_factory_prompt(path)


if __name__ == "__main__":
    main()
