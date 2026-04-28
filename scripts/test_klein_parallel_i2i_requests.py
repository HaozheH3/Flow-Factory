#!/usr/bin/env python3
"""
Parallel I2I load test for Klein: ``multi_modal_data`` uses **only absolute local
paths** (Klein server does not fetch http(s) or decode base64 for condition images).

``generation_params.json`` often lists ``reference_images`` as URLs. This script
resolves each URL to ``local_path`` by scanning ``reference_selection*.json`` in the
same ``results/<run_id>/`` directory (``candidate_image_mappings`` / ``selected_image``).
If any URL cannot be resolved to an existing file, that example is **skipped**.

``output_path`` in each POST is a meaningful local path under ``KLEIN_TEST_OUTPUT_DIR``
(or a default run directory under this repo).

While requests complete, a **tqdm** bar shows overall progress; each finished request prints
a block (via ``tqdm.write``) with ``original_run_folder``, ``generation_params.json`` path,
each original ``reference_images`` entry paired with the **local path** sent to the API,
and ``target_image_path`` (the ``output_path`` used in that POST).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_PRODUCTION_ROOT = Path(
    "/primus_xpfs_workspace_T04/haozhe/ToolGen/phase4_agent/"
    "production_searchbetter_top500_sft_qw1_debug"
)


def _slug_for_path(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip()).strip("_").lower()
    return (t[:max_len].rstrip("_") or "example") if t else "example"


def _default_output_base() -> Path:
    repo = Path(__file__).resolve().parents[1]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return repo / "scripts" / "test_klein_server_outputs" / f"i2i_run_{stamp}"


def _url_match_core(a: str, b: str) -> bool:
    pa, pb = urlparse(a.strip()), urlparse(b.strip())
    return (
        pa.scheme.lower() == pb.scheme.lower()
        and pa.netloc.lower() == pb.netloc.lower()
        and pa.path.rstrip("/").lower() == pb.path.rstrip("/").lower()
    )


def _resolve_local_path_string(lp: str, result_dir: Path, examples_root: Path) -> str | None:
    """Resolve ``local_path`` from selection JSON to an existing absolute path."""
    lp = lp.strip()
    if not lp:
        return None
    p = Path(lp)
    if p.is_absolute() and p.is_file():
        return str(p.resolve())
    anchors = [
        result_dir,
        examples_root,
        examples_root.parent,
        examples_root.parent.parent,
        Path("/primus_xpfs_workspace_T04/haozhe"),
    ]
    for anchor in anchors:
        cand = (anchor / lp).resolve()
        if cand.is_file():
            return str(cand)
    return None


def _build_url_path_pairs(result_dir: Path, examples_root: Path) -> list[tuple[str, str]]:
    """(remote_url, resolved_abs_path) from all ``reference_selection*.json``."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sel in sorted(result_dir.glob("reference_selection*.json")):
        try:
            data = json.loads(sel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        blocks: list[list[dict[str, Any]]] = []
        m = data.get("candidate_image_mappings")
        if isinstance(m, list):
            blocks.append(m)
        si = data.get("selected_image")
        if isinstance(si, dict):
            blocks.append([si])
        for block in blocks:
            for item in block:
                if not isinstance(item, dict):
                    continue
                u = item.get("url") or item.get("imageUrl")
                lp = item.get("local_path")
                if not isinstance(u, str) or not isinstance(lp, str):
                    continue
                resolved = _resolve_local_path_string(lp, result_dir, examples_root)
                if not resolved:
                    continue
                key = (u.strip(), resolved)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((u.strip(), resolved))
    return pairs


def _lookup_local_for_url(url: str, pairs: list[tuple[str, str]]) -> str | None:
    u = url.strip()
    for pu, path in pairs:
        if pu == u:
            return path
    for pu, path in pairs:
        if _url_match_core(pu, u):
            return path
    return None


def _resolve_reference_list_to_local_paths(
    ref_list: list[Any],
    result_dir: Path,
    examples_root: Path,
) -> list[str] | None:
    """Map ``generation_params.json`` ``reference_images`` to existing local paths."""
    index = _build_url_path_pairs(result_dir, examples_root)
    out: list[str] = []
    for ref in ref_list:
        if not isinstance(ref, str) or not ref.strip():
            return None
        r = ref.strip()
        if r.startswith("http://") or r.startswith("https://"):
            hit = _lookup_local_for_url(r, index)
            if not hit:
                return None
            out.append(hit)
            continue
        resolved = _resolve_local_path_string(r, result_dir, examples_root)
        if not resolved:
            p = Path(r).expanduser()
            resolved = str(p.resolve()) if p.is_file() else None
        if not resolved:
            return None
        out.append(resolved)
    return out


def discover_i2i_examples(
    examples_root: Path,
    *,
    min_refs: int,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not examples_root.is_dir():
        return out
    for gp in sorted(examples_root.glob("results/*/generation_params.json")):
        if len(out) >= limit:
            break
        result_dir = gp.parent
        try:
            data = json.loads(gp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        prompt = data.get("prompt") or data.get("prompts")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        refs = data.get("reference_images") or []
        if not isinstance(refs, list) or len(refs) < min_refs:
            continue
        local_paths = _resolve_reference_list_to_local_paths(refs, result_dir, examples_root)
        if local_paths is None or len(local_paths) < min_refs:
            continue
        out.append(
            {
                "run_id": result_dir.name,
                "run_folder": str(result_dir.resolve()),
                "source_file": str(gp),
                "prompt": prompt.strip(),
                "reference_images_original": [str(x) for x in refs if isinstance(x, str)],
                "local_reference_paths": local_paths,
            }
        )
    return out


def _build_multi_modal_local_paths(paths: list[str]) -> list[dict[str, str]]:
    import mimetypes

    mmd: list[dict[str, str]] = []
    for p in paths:
        mt = mimetypes.guess_type(p)[0] or "image/png"
        mmd.append({"mime_type": mt, "content": p})
    return mmd


def _parse_sse_content(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if line.startswith("data:"):
            raw = line[len("data:") :].strip()
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
    return None


def one_i2i_request(
    base: str,
    idx: int,
    example: dict[str, Any],
    output_path: str | None,
    timeout: int,
) -> dict[str, Any]:
    url = base.rstrip("/") + "/api"
    body: dict[str, Any] = {
        "context_request_id": str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "prompts": example["prompt"],
        "size": "1024*1024",
        "seed": (idx * 4243 + 7) % 1_000_000,
        "adapter_id": "flux2-klein-i2i-loadtest",
        "multi_modal_data": _build_multi_modal_local_paths(example["local_reference_paths"]),
    }
    if output_path:
        body["output_path"] = output_path

    row: dict[str, Any] = {
        "idx": idx,
        "run_id": example.get("run_id", ""),
        "run_folder": example.get("run_folder", ""),
        "ok": False,
        "post_url": url,
        "n_refs": len(example["local_reference_paths"]),
        "request_output_path": output_path or "",
        "source_file": example.get("source_file", ""),
        "refs_original": list(example.get("reference_images_original") or []),
        "refs_local": example["local_reference_paths"],
        "prompt_preview": (example["prompt"][:160] + "…") if len(example["prompt"]) > 160 else example["prompt"],
        "result_ref": "",
        "artifact_bytes": 0,
        "error": "",
    }
    try:
        r = requests.post(url, json=body, timeout=timeout)
        if r.status_code != 200:
            row["error"] = f"http {r.status_code} {r.text[:300]}"
            return row
        data = _parse_sse_content(r.text)
        if not data:
            row["error"] = "no data: line in body"
            return row
        if data.get("code") != 0:
            row["error"] = f"code={data.get('code')} msg={data.get('message')}"
            return row
        content = (data.get("data") or {}).get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            row["error"] = "empty content"
            return row
        row["result_ref"] = str(content).strip()
        if row["result_ref"].startswith("http"):
            g = requests.get(row["result_ref"], timeout=timeout)
            row["artifact_bytes"] = len(g.content or b"")
            if g.status_code != 200 or row["artifact_bytes"] < 100:
                row["error"] = f"artifact GET {g.status_code} len={row['artifact_bytes']}"
                return row
        row["ok"] = True
        return row
    except Exception as e:
        row["error"] = str(e)
        return row


def _log_consistency_line(row: dict[str, Any], *, use_tqdm_write: bool) -> None:
    """Print original run folder, source JSON, URL→path mapping, and target path."""
    lines = [
        "",
        f"[{'OK' if row.get('ok') else 'FAIL'}] req={row.get('idx', -1):04d}",
        f"  original_run_folder:     {row.get('run_folder') or '(unknown)'}",
        f"  generation_params.json:  {row.get('source_file') or ''}",
        f"  target_image_path:       {row.get('request_output_path') or '(none in JSON)'}",
        f"  result_ref (API):        {row.get('result_ref') or ''}",
    ]
    orig = row.get("refs_original") or []
    loc = row.get("refs_local") or []
    lines.append("  reference_images (generation_params order) → local paths sent:")
    for i, (o, l) in enumerate(zip(orig, loc), start=1):
        lines.append(f"    [{i}] original: {o}")
        lines.append(f"        local:    {l}")
    if len(orig) != len(loc):
        lines.append(f"    (mismatch count orig={len(orig)} local={len(loc)})")
    if row.get("artifact_bytes"):
        lines.append(f"  artifact_bytes (GET):    {row['artifact_bytes']}")
    if not row.get("ok") and row.get("error"):
        lines.append(f"  error: {row['error'][:400]}")
    lines.append("  " + "-" * 100)
    text = "\n".join(lines)
    if use_tqdm_write:
        try:
            from tqdm import tqdm

            tqdm.write(text)
        except Exception:
            print(text)
    else:
        print(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel Klein I2I: local paths only, resolved from reference_selection*.json")
    p.add_argument("ip", help="Server IP (gateway)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    p.add_argument("--examples-dir", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    p.add_argument("-n", "--requests", type=int, default=64, dest="n")
    p.add_argument("-j", "--jobs", type=int, default=8)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--min-refs", type=int, default=2)
    p.add_argument("--no-output-path", action="store_true")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (still prints consistency lines after all finish).",
    )
    args = p.parse_args()

    examples = discover_i2i_examples(
        args.examples_dir,
        min_refs=args.min_refs,
        limit=args.n,
    )
    if len(examples) < args.n:
        print(
            f"ERROR: only {len(examples)} examples resolved (need {args.n}). "
            "URLs must appear in reference_selection*.json with a resolvable local_path file.",
            file=sys.stderr,
        )
        sys.exit(2)

    n = len(examples)
    output_paths: list[str | None]
    out_base: Path | None = None
    if args.no_output_path:
        output_paths = [None] * n
    else:
        raw = (os.environ.get("KLEIN_TEST_OUTPUT_DIR") or "").strip()
        out_base = Path(raw).expanduser() if raw else _default_output_base()
        out_base.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for i, ex in enumerate(examples):
            slug = _slug_for_path(f"{ex['run_id']}_{ex['prompt'][:40]}")
            output_paths.append(str(out_base / f"i2i_{i:02d}_{slug}.png"))

    base = f"http://{args.ip}:{args.port}"
    post_url = base.rstrip("/") + "/api"
    print(f"I2I POST {post_url}  (n={n}, concurrency={args.jobs})")
    print(f"  examples_dir={args.examples_dir}")
    print("  condition images: local paths only (resolved from reference_selection*.json)")
    if not args.no_output_path and out_base is not None:
        print(f"  output_path base: {out_base}")
    print("-" * 120)

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    use_pb = not args.no_progress
    try:
        from tqdm import tqdm as tqdm_cls
    except ImportError:
        tqdm_cls = None  # type: ignore[misc, assignment]
        use_pb = False

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(one_i2i_request, base, i, examples[i], output_paths[i], args.timeout): i
            for i in range(n)
        }
        if use_pb and tqdm_cls is not None:
            for fut in tqdm_cls(as_completed(futs), total=n, desc="I2I POST /api", unit="req"):
                row = fut.result()
                rows.append(row)
                _log_consistency_line(row, use_tqdm_write=True)
        else:
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                if not args.no_progress:
                    _log_consistency_line(row, use_tqdm_write=False)

    rows.sort(key=lambda r: r["idx"])
    ok = sum(1 for r in rows if r["ok"])
    fail = n - ok

    if args.no_progress:
        for r in rows:
            _log_consistency_line(r, use_tqdm_write=False)

    print("-" * 120)
    print("Summary (by req idx):")
    for r in rows:
        st = "ok" if r["ok"] else "FAIL"
        print(
            f"  {r['idx']:04d} {st}  run={r.get('run_id','')[:24]:<24}  "
            f"target={r.get('request_output_path','')[:60]}"
        )
    print("-" * 120)
    print(f"done in {time.perf_counter() - t0:.1f}s  ok={ok}  fail={fail}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
