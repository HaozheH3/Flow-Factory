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
Merge base Flow-Factory i2i JSONL lines with ToolGen-style eval fields for ``toolgen_searchbetter_judge``.

Each base line must include ``trajectory_id`` (or the key you pass via ``--join-key``).
Each meta line must include the same join key plus ``user_prompt``, ``verification_checklist``,
and ``evaluation_rubric`` (dict or JSON string of a dict). Output lines store ``evaluation_rubric``
as a JSON string for Arrow-stable JSONL loading. Optional ``augmented_generation_details`` is
copied when present.

Example:
  python scripts/merge_toolgen_flow_factory_i2i_jsonl.py \\
    --base-jsonl dataset/my_i2i/base.jsonl \\
    --meta-jsonl /path/to/toolgen_request_meta.jsonl \\
    --output dataset/my_i2i/merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"{path}: line {line_no}: expected JSON object, got {type(obj).__name__}")
            yield obj


def _load_meta_index(meta_path: Path, join_key: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(meta_path):
        raw_key = obj.get(join_key)
        if raw_key is None:
            raise KeyError(f"meta row missing join key {join_key!r}: keys={list(obj.keys())}")
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"meta row has empty {join_key!r}")
        index[key] = obj
    if not index:
        raise ValueError(f"no rows loaded from meta file {meta_path}")
    return index


def _pick_meta_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in (
        "user_prompt",
        "verification_checklist",
        "evaluation_rubric",
        "augmented_generation_details",
        "request_index",
    ):
        if k in meta:
            out[k] = meta[k]
    return out


def _rubric_dict_from_merged_field(rub: Any, *, trajectory_id: Any) -> Dict[str, Any]:
    if isinstance(rub, dict):
        return rub
    if isinstance(rub, str):
        s = rub.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"evaluation_rubric is not valid JSON (trajectory_id={trajectory_id!r}): {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise TypeError(
                f"evaluation_rubric JSON must decode to a dict (trajectory_id={trajectory_id!r}), "
                f"got {type(parsed).__name__}: {parsed!r}"
            )
        return parsed
    raise TypeError(
        f"evaluation_rubric must be dict or str (trajectory_id={trajectory_id!r}), "
        f"got {type(rub).__name__}: {rub!r}"
    )


def _validate_merged(row: Dict[str, Any]) -> None:
    if not isinstance(row.get("verification_checklist"), list) or not row["verification_checklist"]:
        raise ValueError(
            "merged row missing non-empty verification_checklist; "
            f"trajectory_id={row.get('trajectory_id')!r}"
        )
    rub = _rubric_dict_from_merged_field(
        row.get("evaluation_rubric"), trajectory_id=row.get("trajectory_id")
    )
    if not rub:
        raise ValueError(
            f"merged row missing non-empty evaluation_rubric; trajectory_id={row.get('trajectory_id')!r}"
        )
    row["evaluation_rubric"] = json.dumps(rub, ensure_ascii=False)
    up = row.get("user_prompt")
    if not isinstance(up, str) or not up.strip():
        raise ValueError(f"merged row missing user_prompt string; trajectory_id={row.get('trajectory_id')!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-jsonl", type=Path, required=True)
    p.add_argument("--meta-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--join-key", type=str, default="trajectory_id")
    p.add_argument("--strict", action="store_true", help="Fail if a base row has no meta match.")
    args = p.parse_args()

    meta_index = _load_meta_index(args.meta_jsonl, args.join_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for base in _iter_jsonl(args.base_jsonl):
            raw_key = base.get(args.join_key)
            if raw_key is None:
                raise KeyError(f"base row missing join key {args.join_key!r}: keys={list(base.keys())}")
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"base row has empty {args.join_key!r}")

            meta = meta_index.get(key)
            if meta is None:
                if args.strict:
                    raise KeyError(f"no meta row for {args.join_key}={key!r}")
                skipped += 1
                continue

            merged = dict(base)
            merged.update(_pick_meta_fields(meta))
            _validate_merged(merged)
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {args.output} (skipped {skipped} base rows without meta).")


if __name__ == "__main__":
    main()
