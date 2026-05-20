# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Write evaluation artifacts under ``{save_dir}/{run_name}/{eval_dump_subdir}/...``.

Each eval leaf directory includes ``summary.json``, ``samples.jsonl``, ``images/``, and
``judge_transcripts/`` (one JSON per sample) when judge models populate
``extra_kwargs['_toolgen_judge_transcript']`` (ToolGen / Frontier / vLLM judges do this on every call).

HTTP Klein/Bagel workers respond with SSE ``data: {"code","message","data":{"choices":[...]}}``.
To preserve that nested shape in JSONL (not only the flattened image path), set
``sample.extra_kwargs["generation_api_sse_response"]`` to the parsed payload dict from the HTTP client
(see sibling ``Flow-Factory-bagel-merge`` rollout / generation helpers when using remote workers).

Condition/reference images use ``sample_XXXXX_condition_YY.png`` next to ``sample_XXXXX_generated.png``.
Identical refs across a group dedupe via hard link (or same-dir symlink when hard link fails). JSONL rows
list paths in ``condition_image_files``.
"""

from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..samples import BaseSample
from ..utils.image import standardize_image_batch
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# Parsed JSON object from the first ``data:`` SSE line of a Klein/Bagel/ToolGen-compatible
# generation POST response (same structure servers build in ``_sse_body``).
GENERATION_API_SSE_RESPONSE_KEY = "generation_api_sse_response"

# Mirrors ``TOOLGEN_JUDGE_TRANSCRIPT_KEY`` in ``rewards/toolgen_searchbetter_judge_common``;
# inlined so ``utils.eval_dump`` does not import ``flow_factory.rewards`` (avoids fragile import graphs).
_EXTRA_KW_TOOLGEN_JUDGE_TRANSCRIPT = "_toolgen_judge_transcript"


def _has_dataclass_field(sample: BaseSample, name: str) -> bool:
    return name in getattr(type(sample), "__dataclass_fields__", {})


def _save_tensor_or_pil_image(img: Any, path: str) -> None:
    pil_batch = standardize_image_batch(img, output_type="pil")
    if isinstance(pil_batch, list):
        pil = pil_batch[0]
    else:
        pil = pil_batch
    pil.save(path)


def _write_or_link_condition_png(
    *,
    pil: Image.Image,
    digest_to_canonical_abs: Dict[str, str],
    duplicate_abs: str,
    sample_index: int,
    slot_index: int,
) -> str:
    """
    Persist ``duplicate_abs`` under ``images/sample_{i}_condition_{slot}.png``: first digest writes bytes;
    later identical PNGs hardlink (or symlink) to the canonical file. Returns POSIX path relative to run root.
    """
    buf = BytesIO()
    pil.save(buf, format="PNG")
    raw = buf.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    rel = f"images/sample_{sample_index:05d}_condition_{slot_index:02d}.png"

    canonical_abs = digest_to_canonical_abs.get(digest)
    if canonical_abs is None:
        with open(duplicate_abs, "wb") as fh:
            fh.write(raw)
        digest_to_canonical_abs[digest] = os.path.abspath(duplicate_abs)
        return rel

    source = os.path.abspath(canonical_abs)
    if os.path.exists(duplicate_abs):
        try:
            with open(duplicate_abs, "rb") as fh:
                existing_digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError as exc:
            raise OSError(
                f"could not read existing condition image at {duplicate_abs!r} "
                f"(sample_index={sample_index} slot={slot_index})"
            ) from exc
        if existing_digest == digest:
            return rel
        os.remove(duplicate_abs)
    try:
        os.link(source, duplicate_abs)
    except OSError as exc_hard:
        base = os.path.basename(source)
        try:
            os.symlink(base, duplicate_abs)
        except OSError as exc_sym:
            raise OSError(
                f"duplicate condition image for sample_index={sample_index} slot={slot_index}: "
                f"could not hardlink {duplicate_abs!r} to {source!r} ({exc_hard!r}); "
                f"same-directory symlink also failed ({exc_sym!r})."
            ) from exc_sym
    return rel


def dump_eval_artifacts(
    *,
    output_dir: str,
    epoch: int,
    step: int,
    samples: List[BaseSample],
    gathered_rewards: Dict[str, np.ndarray],
    status: str,
    error_traceback: Optional[str],
) -> Optional[str]:
    """
    Persist eval summary, per-sample JSONL metadata, rasterized images, and optional judge transcripts.

    Rows with conditioning include ``condition_image_files``: paths
    ``images/sample_*_condition_*.png`` alongside ``images/sample_*_generated.png``; identical bytes use
    hard links or same-directory symlinks.

    Returns:
        ``output_dir`` if anything was written, else ``None``.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    n = len(samples)
    reward_alignment: Dict[str, Any] = {}
    for name, arr in gathered_rewards.items():
        reward_alignment[name] = {
            "shape": list(arr.shape),
            "aligned_with_samples": bool(arr.shape[0] == n),
        }

    summary: Dict[str, Any] = {
        "epoch": int(epoch),
        "step": int(step),
        "status": status,
        "num_samples": n,
        "reward_alignment": reward_alignment,
        "reward_stats": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for k, v in gathered_rewards.items()
            if v.size > 0
        },
    }
    if error_traceback:
        summary["error_traceback"] = error_traceback

    jsonl_path = os.path.join(output_dir, "samples.jsonl")
    judge_rel_root = "judge_transcripts"
    judge_abs_root = os.path.join(output_dir, judge_rel_root)
    judge_written = False
    png_digest_to_canonical_abs: Dict[str, str] = {}
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, sample in enumerate(samples):
            row: Dict[str, Any] = {
                "index": i,
                "sample_type": type(sample).__name__,
                "prompt": sample.prompt,
                "generated_image_file": f"images/sample_{i:05d}_generated.png",
            }
            for meta_key in ("trajectory_id", "request_index"):
                if meta_key in sample.extra_kwargs:
                    row[meta_key] = sample.extra_kwargs[meta_key]
                elif _has_dataclass_field(sample, meta_key):
                    row[meta_key] = getattr(sample, meta_key, None)

            for rk, rv in gathered_rewards.items():
                if rv.shape[0] == n:
                    row[f"reward_{rk}"] = float(rv[i])

            # Compact judge / ToolGen extras when present
            for ek in (
                "user_prompt",
                "reward_judge",
                "_toolgen_judge_labeled_scores",
            ):
                if ek in sample.extra_kwargs:
                    val = sample.extra_kwargs[ek]
                    if ek == "_toolgen_judge_labeled_scores" and isinstance(val, list):
                        row[ek] = val[:50]
                        if len(val) > 50:
                            row[f"{ek}_truncated"] = True
                    else:
                        row[ek] = val

            sse_payload = sample.extra_kwargs.get(GENERATION_API_SSE_RESPONSE_KEY)
            if sse_payload is None:
                sse_payload = sample.extra_kwargs.get("model_raw_sse_response")
            if sse_payload is not None:
                row[GENERATION_API_SSE_RESPONSE_KEY] = sse_payload

            transcript = sample.extra_kwargs.get(_EXTRA_KW_TOOLGEN_JUDGE_TRANSCRIPT)
            if transcript is not None:
                if not judge_written:
                    os.makedirs(judge_abs_root, exist_ok=True)
                    judge_written = True
                judge_name = f"sample_{i:05d}.json"
                judge_path = os.path.join(judge_abs_root, judge_name)
                row["judge_transcript_file"] = f"{judge_rel_root}/{judge_name}"
                with open(judge_path, "w", encoding="utf-8") as wf:
                    json.dump(transcript, wf, indent=2, ensure_ascii=False, default=str)

            if _has_dataclass_field(sample, "condition_images"):
                cond = getattr(sample, "condition_images", None)
                if cond is not None:
                    cond_pils = standardize_image_batch(cond, output_type="pil")
                    if not isinstance(cond_pils, list):
                        cond_pils = [cond_pils]
                    cond_paths: List[str] = []
                    for j, one in enumerate(cond_pils):
                        if not isinstance(one, Image.Image):
                            raise TypeError(
                                f"expected PIL.Image.Image in condition_images for sample index {i}; "
                                f"got {type(one).__name__}: {one!r}"
                            )
                        abs_cond = os.path.join(
                            img_dir, f"sample_{i:05d}_condition_{j:02d}.png"
                        )
                        cond_paths.append(
                            _write_or_link_condition_png(
                                pil=one,
                                digest_to_canonical_abs=png_digest_to_canonical_abs,
                                duplicate_abs=abs_cond,
                                sample_index=i,
                                slot_index=j,
                            )
                        )
                    row["condition_image_files"] = cond_paths

            jf.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

            if sample.image is not None:
                _save_tensor_or_pil_image(
                    sample.image,
                    os.path.join(output_dir, row["generated_image_file"]),
                )

    summary["condition_images_unique_png_written"] = int(len(png_digest_to_canonical_abs))
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)

    logger.info("Wrote evaluation artifacts to %s (status=%s)", output_dir, status)
    return output_dir
