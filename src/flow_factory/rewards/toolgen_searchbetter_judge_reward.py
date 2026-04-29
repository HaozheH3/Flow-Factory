# Copyright 2026 Jayce-Ping, Haozhe Wang
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
ToolGen SearchBetter-style multimodal judge rewards.

- ``ToolGenSearchBetterJudgeRewardModel``: OpenAI-compatible HTTP (AsyncOpenAI).
- ``ToolGenSearchBetterJudgeFrontierRewardModel``: Alibaba Frontier / llm-chat-api via ToolGen
  ``frontier_model.FrontierModel`` (see ``toolgen_searchbetter_judge_common.import_toolgen_frontier_model``).

Prompt text and interleaved image layout match ``evaluate_searchbetter_hard_direct.py`` and saved
``*_prompt_context.txt`` files from phase5 evaluation runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from .toolgen_searchbetter_judge_common import (
    JUDGE_SYSTEM_PROMPT,
    as_rubric_dict,
    as_str_list,
    build_eval_prompt_text,
    build_judge_interleaved_user_content,
    build_visual_reference_context_from_row,
    build_visual_reference_context_minimal,
    coerce_ref_image_list,
    import_toolgen_frontier_model,
    parse_toolgen_judge_major_aspect_weights_cfg,
    parsed_judge_json_to_weighted_reward,
    parsed_judge_labeled_scores_03,
    reference_slot_placeholders,
    resolve_toolgen_judge_unparsed_response_dump_dir,
    try_parse_judge_output,
    validate_meta_list,
    write_toolgen_judge_unparsed_response_dump,
)
from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64

import urllib3

# Frontier / llm-chat-api client may use unverified TLS; avoid spamming training logs.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

FRONTIER_JUDGE_LOG = logging.getLogger(__name__ + ".frontier_judge")
logger = logging.getLogger(__name__)

# One full judge completion plus one extra completion if the response does not parse.
TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES = 2


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... truncated, original_len={len(text)}"


def _redact_interleaved_for_dump(parts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop base64 payloads from multimodal parts for on-disk JSON dumps."""
    out: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            out.append({"repr": repr(part)[:200]})
            continue
        if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
            url = part["image_url"].get("url", "")
            u = str(url)
            redacted = (
                f"<redacted data URL; chars={len(u)}>"
                if u.startswith("data:")
                else _truncate_text(u, 200)
            )
            clone = dict(part)
            clone["image_url"] = {**part["image_url"], "url": redacted}
            out.append(clone)
        else:
            out.append(part)
    return out


def _summarize_api_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return {"type": type(result).__name__}
    if "_error" in result:
        err = result.get("_error")
        return {"_error": err if isinstance(err, str) else repr(err)}
    summary: Dict[str, Any] = {"keys": sorted(result.keys())}
    ch = result.get("choices")
    if isinstance(ch, list):
        summary["num_choices"] = len(ch)
    return summary


def _filesystem_safe_stem_component(raw: str, max_len: int) -> str:
    """Compress an id into a short, filename-safe fragment (may be empty)."""
    s = str(raw).replace(os.sep, "_").replace("/", "_").replace("\\", "_").strip()
    if not s:
        return ""
    out: List[str] = []
    for ch in s:
        if len(out) >= max_len:
            break
        if ch.isalnum() or ch in "._-+@":
            out.append(ch)
        else:
            out.append("_")
    compact = "".join(out).strip("_")
    return compact[:max_len]


def _make_judge_response_dump_stem(
    *,
    run_id: str,
    intra_batch_index: int,
    trajectory_id: str,
    request_index: Any,
    suffix: str = "unparsed",
) -> str:
    """Unique basename fragment for judge artifact files (unparsed dumps, etc.)."""
    rid = str(run_id).strip() or uuid.uuid4().hex[:12]
    traj_part = _filesystem_safe_stem_component(str(trajectory_id or ""), 40)
    req_part = _filesystem_safe_stem_component(str(request_index), 24)
    tail = f"{suffix}_b{intra_batch_index:04d}_{time.time_ns()}"
    if traj_part and req_part:
        return f"{rid}__{traj_part}__r{req_part}__{tail}"
    if traj_part:
        return f"{rid}__{traj_part}__{tail}"
    return f"{rid}__{tail}"


class ToolGenSearchBetterJudgeRewardModel(PointwiseRewardModel):
    """
    Pointwise i2i reward via remote multimodal judge (OpenAI-compatible chat completions).

    ``extra_kwargs`` (subset): ``api_base_url``, ``api_key``, ``vlm_model``, ``max_concurrent``,
    ``max_retries``, ``timeout``, ``temperature``, ``max_tokens``, ``max_pixels`` (per image_url,
    default ``589824``), ``variant`` (string tag in the evaluation context, default ``training``).
    After transport retries, **one** extra full judge completion is attempted when the reply is
    empty, unparsable as XML/JSON, or fails labeled-score extraction; only then errors propagate.
    ``toolgen_judge_unparsed_dump_dir`` (optional path): when the judge reply cannot be parsed as
    XML/JSON, write the full raw assistant string to a ``*.judge_unparsed.txt`` file under this
    directory (see also env ``FLOW_FACTORY_JUDGE_UNPARSED_DUMP_DIR`` / ``TOOLGEN_JUDGE_UNPARSED_DUMP_DIR``).
    Aggregated reward (see :func:`flow_factory.rewards.toolgen_searchbetter_judge_common.parsed_judge_json_to_weighted_reward`):
    ``toolgen_judge_checklist_item_weight`` (default ``1``) for **checklist** sub-aspect renormalization;
    ``toolgen_judge_major_aspect_weights`` (optional dict) for renormalized weights across **major** terms
    (adaptive rubric block, checklist block, each generic rubric dimension, reference scores).
    Group-level scoring (optional): ``toolgen_group_drop_constant_score_dims`` (bool), ``toolgen_group_constant_score_std_eps``,
    ``toolgen_retain_labeled_scores_in_samples`` — see :class:`ToolGenSearchBetterJudgeFrontierRewardModel` docstring.
    """

    required_fields = (
        "prompt",
        "image",
        "condition_images",
        "user_prompt",
        "verification_checklist",
        "evaluation_rubric",
    )
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "ToolGenSearchBetterJudgeRewardModel requires the `openai` package. "
                "Install with: pip install openai"
            ) from e

        self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8000/v1")
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get("vlm_model", "gpt-4o-mini")
        self.max_concurrent = int(config.extra_kwargs.get("max_concurrent", 4))
        self.max_retries = int(config.extra_kwargs.get("max_retries", 3))
        self.timeout = float(config.extra_kwargs.get("timeout", 180.0))
        self.temperature = float(config.extra_kwargs.get("temperature", 0.2))
        self.max_tokens = int(config.extra_kwargs.get("max_tokens", 8000))
        self.max_pixels = int(config.extra_kwargs.get("max_pixels", 589824))
        self.variant = str(config.extra_kwargs.get("variant", "training"))
        self._judge_reward_weights = {
            "checklist_item_weight": float(
                config.extra_kwargs.get("toolgen_judge_checklist_item_weight", 1.0)
            ),
            "major_aspect_weights": parse_toolgen_judge_major_aspect_weights_cfg(
                config.extra_kwargs.get("toolgen_judge_major_aspect_weights")
            ),
        }

        self.client = AsyncOpenAI(base_url=self.api_base_url, api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max(1, self.max_concurrent))

        self._unparsed_dump_dir = resolve_toolgen_judge_unparsed_response_dump_dir(config.extra_kwargs)
        if self._unparsed_dump_dir is not None:
            self._unparsed_dump_dir.mkdir(parents=True, exist_ok=True)
            env_run = os.environ.get("FLOW_FACTORY_JUDGE_RUN_ID")
            self._judge_artifact_run_id = (
                str(env_run).strip() if isinstance(env_run, str) and env_run.strip() else uuid.uuid4().hex[:12]
            )
        else:
            self._judge_artifact_run_id = ""

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Any]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        user_prompt: Optional[List[str]] = None,
        verification_checklist: Optional[List[Any]] = None,
        evaluation_rubric: Optional[List[Any]] = None,
        augmented_generation_details: Optional[List[Any]] = None,
        trajectory_id: Optional[List[Any]] = None,
        request_index: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> RewardModelOutput:
        del kwargs
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided for ToolGenSearchBetterJudge")
        if condition_images is None:
            raise ValueError("condition_images is required for ToolGenSearchBetterJudge")
        if user_prompt is None:
            raise ValueError(
                "user_prompt is required (store on each sample, e.g. via extra_kwargs from JSONL)."
            )

        batch_len = len(prompt)
        if len(image) != batch_len or len(condition_images) != batch_len:
            raise ValueError(
                f"expected len(prompt)==len(image)==len(condition_images)==batch_size, got "
                f"{len(prompt)}, {len(image)}, {len(condition_images)}"
            )
        validate_meta_list("user_prompt", user_prompt, batch_len)
        for i, up in enumerate(user_prompt):
            if up is None:
                raise ValueError(
                    f"sample {i}: user_prompt is required but got None. "
                    "Store user_prompt on every sample (top-level or extra_kwargs from JSONL)."
                )
        if verification_checklist is None:
            verification_checklist = [[] for _ in range(batch_len)]
        else:
            validate_meta_list("verification_checklist", verification_checklist, batch_len)
        if evaluation_rubric is None:
            evaluation_rubric = [{} for _ in range(batch_len)]
        else:
            validate_meta_list("evaluation_rubric", evaluation_rubric, batch_len)

        if augmented_generation_details is None:
            augmented_generation_details = [None] * batch_len
        else:
            validate_meta_list("augmented_generation_details", augmented_generation_details, batch_len)

        if trajectory_id is None:
            trajectory_id = [""] * batch_len
        else:
            validate_meta_list("trajectory_id", trajectory_id, batch_len)

        if request_index is None:
            request_index = [-1] * batch_len
        else:
            validate_meta_list("request_index", request_index, batch_len)

        ref_lists = [coerce_ref_image_list(c) for c in condition_images]
        meta_checklists = [as_str_list(verification_checklist[i], sample_index=i) for i in range(batch_len)]
        meta_rubrics = [as_rubric_dict(evaluation_rubric[i], sample_index=i) for i in range(batch_len)]

        scores, labeled_batch = asyncio.run(
            self._async_score_batch(
                user_prompts=[str(user_prompt[i]) for i in range(batch_len)],
                edited=image,
                ref_lists=ref_lists,
                checklists=meta_checklists,
                rubrics=meta_rubrics,
                aug_details=list(augmented_generation_details),
                traj_ids=[str(trajectory_id[i] or "") for i in range(batch_len)],
                req_indices=[request_index[i] for i in range(batch_len)],
            )
        )
        rewards = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return RewardModelOutput(
            rewards=rewards,
            extra_info={"judge_labeled_scores_batch": labeled_batch},
        )

    async def _async_score_batch(
        self,
        user_prompts: List[str],
        edited: List[Image.Image],
        ref_lists: List[List[Image.Image]],
        checklists: List[List[str]],
        rubrics: List[Dict[str, Any]],
        aug_details: List[Any],
        traj_ids: List[str],
        req_indices: List[Any],
    ) -> tuple[List[float], List[List[tuple[str, float]]]]:
        tasks = [
            self._score_single(
                intra_batch_index=i,
                user_prompt=user_prompts[i],
                edited=edited[i],
                refs=ref_lists[i],
                checklist=checklists[i],
                rubric=rubrics[i],
                aug_detail=aug_details[i],
                trajectory_id=traj_ids[i],
                request_index=req_indices[i],
            )
            for i in range(len(user_prompts))
        ]
        pairs = list(await asyncio.gather(*tasks))
        scores = [p[0] for p in pairs]
        labeled = [p[1] for p in pairs]
        return scores, labeled

    async def _score_single(
        self,
        *,
        intra_batch_index: int,
        user_prompt: str,
        edited: Image.Image,
        refs: List[Image.Image],
        checklist: List[str],
        rubric: Dict[str, Any],
        aug_detail: Any,
        trajectory_id: str,
        request_index: Any,
    ) -> tuple[float, List[tuple[str, float]]]:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        slot_urls = reference_slot_placeholders(len(refs))
        ref_data_urls = [pil_image_to_base64(r, format="PNG") for r in refs]
        assess_url = pil_image_to_base64(edited, format="PNG")

        row: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "trajectory_id": trajectory_id,
            "request_index": request_index,
        }
        if isinstance(aug_detail, dict):
            row["augmented_generation_details"] = aug_detail
            visual_context = build_visual_reference_context_from_row(row)
        else:
            visual_context = build_visual_reference_context_minimal(len(refs) > 0)

        text_prompt = build_eval_prompt_text(
            row=row,
            verification_checklist=checklist,
            evaluation_rubric=rubric,
            visual_context=visual_context,
            variant=self.variant,
            reference_slot_urls=slot_urls,
        )
        interleaved = build_judge_interleaved_user_content(
            user_text_prompt=text_prompt,
            reference_slot_urls=slot_urls,
            reference_image_urls=ref_data_urls,
            visual_context=visual_context,
            assess_image_data_url=assess_url,
            max_pixels=self.max_pixels,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": interleaved},
        ]

        last_transport_err: Optional[BaseException] = None
        for parse_try in range(TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES):
            completion = None
            last_transport_err = None
            for attempt in range(self.max_retries):
                try:
                    async with self.semaphore:
                        completion = await self.client.chat.completions.create(
                            model=self.vlm_model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            timeout=self.timeout,
                        )
                    break
                except (APIConnectionError, APITimeoutError, RateLimitError, asyncio.TimeoutError) as e:
                    last_transport_err = e
                    if attempt + 1 >= self.max_retries:
                        break
                    await asyncio.sleep(2**attempt)
            if completion is None:
                raise RuntimeError(
                    f"ToolGenSearchBetterJudge HTTP request failed after {self.max_retries} attempt(s)"
                ) from last_transport_err

            content = completion.choices[0].message.content
            if content is None or not str(content).strip():
                if parse_try + 1 < TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES:
                    logger.warning(
                        "ToolGenSearchBetterJudge: empty judge content (parse_try %s/%s); retrying completion",
                        parse_try + 1,
                        TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES,
                    )
                    continue
                raise ValueError(
                    "Judge returned empty assistant content; cannot parse ToolGen-style judge output"
                )

            raw = str(content)
            parsed = try_parse_judge_output(raw)
            if parsed is None:
                if parse_try + 1 < TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES:
                    logger.warning(
                        "ToolGenSearchBetterJudge: failed to parse judge output (parse_try %s/%s); retrying completion",
                        parse_try + 1,
                        TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES,
                    )
                    continue
                if self._unparsed_dump_dir is not None:
                    stem = _make_judge_response_dump_stem(
                        run_id=self._judge_artifact_run_id,
                        intra_batch_index=intra_batch_index,
                        trajectory_id=str(trajectory_id or ""),
                        request_index=request_index,
                    )
                    write_toolgen_judge_unparsed_response_dump(
                        self._unparsed_dump_dir,
                        stem=stem,
                        raw_response=raw,
                        meta={
                            "reward_model": "toolgen_searchbetter_judge",
                            "vlm_model": self.vlm_model,
                            "variant": self.variant,
                            "status": "parse_error",
                        },
                    )
                raise ValueError(
                    "Failed to parse strict XML/JSON from judge response; "
                    f"first 500 chars: {raw[:500]!r}"
                )

            try:
                labeled = parsed_judge_labeled_scores_03(parsed)
                reward = float(
                    parsed_judge_json_to_weighted_reward(
                        parsed, rubric, **self._judge_reward_weights
                    )
                )
            except (ValueError, TypeError, KeyError) as exc:
                if parse_try + 1 < TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES:
                    logger.warning(
                        "ToolGenSearchBetterJudge: judge labeled scores invalid (%s; parse_try %s/%s); retrying completion",
                        exc,
                        parse_try + 1,
                        TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES,
                    )
                    continue
                raise

            return reward, labeled

        raise RuntimeError("ToolGenSearchBetterJudge: exhausted parse completion retries without success")


class ToolGenSearchBetterJudgeFrontierRewardModel(PointwiseRewardModel):
    """
    Same judge prompt as :class:`ToolGenSearchBetterJudgeRewardModel`, but calls Alibaba Frontier
    ``llm-chat-api`` through ToolGen's ``FrontierModel`` (synchronous ``requests`` backend).

    Requires a checkout of ToolGen on disk. Set ``extra_kwargs``:

    - ``toolgen_phase4_agent_dir`` (required unless env ``TOOLGEN_PHASE4_AGENT_DIR`` is set): absolute
      path to ``.../ToolGen/phase4_agent`` (the directory containing ``frontier_model.py``).
    - ``frontier_model_name``: model id (default ``doubao-seed-2.0-mini``, same family as phase5 script).
    - ``frontier_api_format``: optional ``llm_chat`` | ``openai`` | ``vllm`` (default: let ``LLMClient`` infer).
    - ``frontier_max_total_pixels``: passed to ``FrontierModel`` (default ``589824``).
    - ``frontier_api_timeout`` or ``timeout`` (int, seconds): HTTP timeout for each Frontier request
      (sets ``LLMClient.default_timeout``; default ToolGen value is ``180``).
    - ``max_concurrent``: thread pool size for parallel judge calls (default ``2``).
    - ``max_retries``, ``temperature``, ``max_tokens``, ``max_pixels``, ``variant``: same semantics as OpenAI path
      (including **one** extra full completion on empty / unparseable / invalid labeled output);
      ``max_pixels`` default ``589824`` (``768 * 768``, matching ``evaluate_searchbetter_hard_direct``).
    - Same aggregate reward args as the HTTP class: ``toolgen_judge_checklist_item_weight``,
      ``toolgen_judge_major_aspect_weights`` (see
      :func:`flow_factory.rewards.toolgen_searchbetter_judge_common.parsed_judge_json_to_weighted_reward`).
    - ``toolgen_judge_unparsed_dump_dir`` (optional path): on XML/JSON parse failure, always write the
      **full** raw assistant string to ``*.judge_unparsed.txt`` (and a small ``*.judge_unparsed_meta.json``)
      under this directory, independent of ``frontier_judge_dump_dir``. Same env vars as
      :class:`ToolGenSearchBetterJudgeRewardModel` (``FLOW_FACTORY_JUDGE_UNPARSED_DUMP_DIR`` /
      ``TOOLGEN_JUDGE_UNPARSED_DUMP_DIR``). When ``frontier_judge_dump_dir`` is also set, the unparsed
      dump reuses the same ``stem`` as the JSON artifact so files correlate by basename.
    - ``frontier_judge_dump_dir`` (optional path): for each judge call, write a pair of files sharing the
      same basename: ``<stem>.json`` (metadata + model text + parsed scores when available) and
      ``<stem>.<image_ext>`` (the generated image passed to the judge). ``run_id`` prefix comes from
      env ``FLOW_FACTORY_JUDGE_RUN_ID`` or a short UUID; ``stem`` also encodes batch slot and timestamp
      for uniqueness. Optional human-readable fragments from ``trajectory_id`` / ``request_index`` are
      included when short enough.
    - ``frontier_judge_dump_image_format``: ``png`` (default), ``webp``, or ``jpeg`` / ``jpg`` (PIL save format).
    - ``frontier_judge_dump_redacted_request`` (bool, default ``True``): store interleaved multimodal
      parts with ``data:`` image URLs redacted (text blocks kept).
    - ``frontier_judge_max_dump_response_chars`` (int, default ``500000``): cap stored assistant text.
    - ``frontier_judge_console_progress`` (bool, default ``True``): ``INFO`` log per sample when a request completes (``DONE``); per-sample ``START`` lines are not emitted to keep logs small.
    - ``frontier_judge_error_policy``: ``raise`` (default) or ``penalize``. After transport retries and
      one extra completion on empty / unparseable / invalid labeled output, ``raise`` aborts training;
      ``penalize`` assigns ``frontier_judge_penalty_value`` (default ``0.0``). When other samples in the
      same ``unique_id`` group have a successful parse, :class:`~flow_factory.rewards.reward_processor.RewardProcessor`
      overwrites the penalty with the mean reward of those successful members before advantage / DPO pairing.
      Successful parses use ``parsed_judge_json_to_weighted_reward``
      (configurable; defaults match the legacy unweighted mean when all weights are ``1``) on ``[0, 1]``.
    - ``toolgen_group_drop_constant_score_dims`` (bool, default ``False``): after scoring, regroup by
      ``sample.unique_id`` and replace each sample's reward with the mean of only those score dimensions
      whose values differ across **successful** judge parses in the group (population std on ``[0,3]`` scores
      ``> toolgen_group_constant_score_std_eps``, default ``1e-6``). Failed parses (no labeled score vector)
      are **omitted** from the std/mask step and receive the **mean** of the adjusted rewards of successful
      members (neutral vs the group mean for advantage-style training). Single-sample groups skip adjustment.
      Successful members must share the same score key ordering (same checklist/rubric layout).
    - ``toolgen_retain_labeled_scores_in_samples`` (bool, default ``False``): if true, keep
      ``extra_kwargs['_toolgen_judge_labeled_scores']`` after adjustment; otherwise it is removed to save memory.
    """

    required_fields = (
        "prompt",
        "image",
        "condition_images",
        "user_prompt",
        "verification_checklist",
        "evaluation_rubric",
    )
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        phase4 = config.extra_kwargs.get("toolgen_phase4_agent_dir") or os.environ.get(
            "TOOLGEN_PHASE4_AGENT_DIR"
        )
        if not phase4 or not isinstance(phase4, str):
            raise ValueError(
                "ToolGenSearchBetterJudgeFrontierRewardModel requires toolgen_phase4_agent_dir in "
                "reward extra_kwargs or TOOLGEN_PHASE4_AGENT_DIR in the environment (path to ToolGen "
                "phase4_agent directory)."
            )
        self.phase4_agent_dir = phase4
        FrontierModel = import_toolgen_frontier_model(self.phase4_agent_dir)

        self.frontier_model_name = str(config.extra_kwargs.get("frontier_model_name", "doubao-seed-2.0-mini"))
        raw_fmt = config.extra_kwargs.get("frontier_api_format")
        self.frontier_api_format: Optional[str] = (
            str(raw_fmt).strip().lower() if isinstance(raw_fmt, str) and raw_fmt.strip() else None
        )
        self.max_total_pixels = int(config.extra_kwargs.get("frontier_max_total_pixels", 589824))
        self.frontier = FrontierModel(
            model_name=self.frontier_model_name,
            temperature=float(config.extra_kwargs.get("temperature", 0.2)),
            api_format=self.frontier_api_format,
            max_total_pixels=self.max_total_pixels,
        )

        self.max_concurrent = int(config.extra_kwargs.get("max_concurrent", 2))
        self.max_retries = int(config.extra_kwargs.get("max_retries", 3))
        self.max_tokens = int(config.extra_kwargs.get("max_tokens", 8000))
        self.max_pixels = int(config.extra_kwargs.get("max_pixels", 589824))
        self.variant = str(config.extra_kwargs.get("variant", "training"))
        self._judge_reward_weights = {
            "checklist_item_weight": float(
                config.extra_kwargs.get("toolgen_judge_checklist_item_weight", 1.0)
            ),
            "major_aspect_weights": parse_toolgen_judge_major_aspect_weights_cfg(
                config.extra_kwargs.get("toolgen_judge_major_aspect_weights")
            ),
        }

        api_timeout = config.extra_kwargs.get("frontier_api_timeout")
        if api_timeout is None:
            api_timeout = config.extra_kwargs.get("timeout")
        if api_timeout is not None:
            self.frontier.client.default_timeout = int(api_timeout)
        self._api_timeout = int(self.frontier.client.default_timeout)

        raw_policy = str(config.extra_kwargs.get("frontier_judge_error_policy", "raise")).strip().lower()
        if raw_policy not in {"raise", "penalize"}:
            raise ValueError(
                f"frontier_judge_error_policy must be 'raise' or 'penalize', got {raw_policy!r}"
            )
        self._error_policy: str = raw_policy
        self._penalty_value = float(config.extra_kwargs.get("frontier_judge_penalty_value", 0.0))

        self._console_progress = bool(config.extra_kwargs.get("frontier_judge_console_progress", True))
        self._dump_redacted_request = bool(
            config.extra_kwargs.get("frontier_judge_dump_redacted_request", True)
        )
        self._max_dump_response_chars = int(
            config.extra_kwargs.get("frontier_judge_max_dump_response_chars", 500_000)
        )

        img_fmt_raw = str(config.extra_kwargs.get("frontier_judge_dump_image_format", "png")).strip().lower()
        if img_fmt_raw in ("jpeg", "jpg"):
            self._dump_pil_image_format = "JPEG"
            self._dump_image_suffix = "jpg"
        elif img_fmt_raw == "webp":
            self._dump_pil_image_format = "WEBP"
            self._dump_image_suffix = "webp"
        elif img_fmt_raw == "png":
            self._dump_pil_image_format = "PNG"
            self._dump_image_suffix = "png"
        else:
            raise ValueError(
                f"frontier_judge_dump_image_format must be png, webp, jpeg, or jpg, got {img_fmt_raw!r}"
            )

        dump_raw = config.extra_kwargs.get("frontier_judge_dump_dir")
        env_run = os.environ.get("FLOW_FACTORY_JUDGE_RUN_ID")
        self._dump_run_id = (
            str(env_run).strip()
            if isinstance(env_run, str) and env_run.strip()
            else uuid.uuid4().hex[:12]
        )
        self._dump_dir: Optional[Path] = None
        self._dump_lock = threading.Lock()
        if dump_raw is not None and str(dump_raw).strip():
            self._dump_dir = Path(str(dump_raw)).expanduser().resolve()
            self._dump_dir.mkdir(parents=True, exist_ok=True)

        self._unparsed_dump_dir = resolve_toolgen_judge_unparsed_response_dump_dir(config.extra_kwargs)
        if self._unparsed_dump_dir is not None:
            self._unparsed_dump_dir.mkdir(parents=True, exist_ok=True)

    def _make_eval_dump_stem(
        self,
        *,
        intra_batch_index: int,
        trajectory_id: str,
        request_index: Any,
    ) -> str:
        traj_part = _filesystem_safe_stem_component(str(trajectory_id or ""), 40)
        req_part = _filesystem_safe_stem_component(str(request_index), 24)
        tail = f"b{intra_batch_index:04d}_{time.time_ns()}"
        if traj_part and req_part:
            return f"{self._dump_run_id}__{traj_part}__r{req_part}__{tail}"
        if traj_part:
            return f"{self._dump_run_id}__{traj_part}__{tail}"
        return f"{self._dump_run_id}__{tail}"

    def _persist_eval_artifact_pair(
        self,
        stem: str,
        record: Dict[str, Any],
        edited: Image.Image,
    ) -> None:
        if self._dump_dir is None:
            return
        if not isinstance(edited, Image.Image):
            raise TypeError(
                f"expected PIL.Image.Image for dump image, got {type(edited).__name__}"
            )
        json_path = self._dump_dir / f"{stem}.json"
        img_path = self._dump_dir / f"{stem}.{self._dump_image_suffix}"
        payload = {
            **record,
            "dump_stem": stem,
            "generated_image_basename": img_path.name,
            "eval_record_basename": json_path.name,
        }
        save_kwargs: Dict[str, Any] = {}
        if self._dump_pil_image_format == "JPEG":
            save_kwargs["quality"] = 95
        to_save = edited
        if edited.mode not in ("RGB", "RGBA", "L"):
            to_save = edited.convert("RGB")
        elif self._dump_pil_image_format == "JPEG" and edited.mode == "RGBA":
            to_save = edited.convert("RGB")
        with self._dump_lock:
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            to_save.save(img_path, format=self._dump_pil_image_format, **save_kwargs)

    def _fail_sample(
        self,
        *,
        base: Dict[str, Any],
        status: str,
        message: str,
        dump_stem: Optional[str],
        edited: Image.Image,
        extra: Optional[Dict[str, Any]] = None,
    ) -> float:
        row = {**base, "status": status, "error": message, "reward": None}
        if extra:
            row.update(extra)
        if dump_stem is not None:
            self._persist_eval_artifact_pair(dump_stem, row, edited)
        FRONTIER_JUDGE_LOG.warning(
            "[%s] status=%s trajectory_id=%r request_index=%r detail=%s",
            base.get("call_id"),
            status,
            base.get("trajectory_id"),
            base.get("request_index"),
            _truncate_text(message, 1500),
        )
        if self._error_policy == "raise":
            if status == "parse_error":
                raise ValueError(message)
            raise RuntimeError(
                f"ToolGenSearchBetterJudgeFrontierRewardModel sample failed ({status}): {message}"
            )
        FRONTIER_JUDGE_LOG.warning(
            "[%s] frontier_judge_error_policy=penalize -> reward=%s",
            base.get("call_id"),
            self._penalty_value,
        )
        return float(self._penalty_value)

    def _score_one_sync(
        self,
        *,
        intra_batch_index: int,
        batch_len: int,
        user_prompt: str,
        edited: Image.Image,
        refs: List[Image.Image],
        checklist: List[str],
        rubric: Dict[str, Any],
        aug_detail: Any,
        trajectory_id: str,
        request_index: Any,
    ) -> tuple[float, Optional[List[tuple[str, float]]]]:
        dump_stem: Optional[str] = None
        if self._dump_dir is not None:
            dump_stem = self._make_eval_dump_stem(
                intra_batch_index=intra_batch_index,
                trajectory_id=trajectory_id,
                request_index=request_index,
            )
        call_id = dump_stem or f"{self._dump_run_id}-b{intra_batch_index}-{time.time_ns()}"
        base = {
            "ts": time.time(),
            "call_id": call_id,
            "intra_batch_index": intra_batch_index,
            "batch_len": batch_len,
            "model_name": self.frontier_model_name,
            "trajectory_id": trajectory_id,
            "request_index": request_index,
            "user_prompt_preview": _truncate_text(str(user_prompt), 800),
            "api_timeout_sec": self._api_timeout,
        }

        slot_urls = reference_slot_placeholders(len(refs))
        ref_data_urls = [pil_image_to_base64(r, format="PNG") for r in refs]
        assess_url = pil_image_to_base64(edited, format="PNG")

        row: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "trajectory_id": trajectory_id,
            "request_index": request_index,
        }
        if isinstance(aug_detail, dict):
            row["augmented_generation_details"] = aug_detail
            visual_context = build_visual_reference_context_from_row(row)
        else:
            visual_context = build_visual_reference_context_minimal(len(refs) > 0)

        text_prompt = build_eval_prompt_text(
            row=row,
            verification_checklist=checklist,
            evaluation_rubric=rubric,
            visual_context=visual_context,
            variant=self.variant,
            reference_slot_urls=slot_urls,
        )
        interleaved = build_judge_interleaved_user_content(
            user_text_prompt=text_prompt,
            reference_slot_urls=slot_urls,
            reference_image_urls=ref_data_urls,
            visual_context=visual_context,
            assess_image_data_url=assess_url,
            max_pixels=self.max_pixels,
        )

        dump_request: Optional[List[Dict[str, Any]]] = None
        if self._dump_dir is not None and self._dump_redacted_request:
            dump_request = _redact_interleaved_for_dump(interleaved)

        for parse_try in range(TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES):
            try:
                result = self.frontier.client.multimodal_chat_multiimages_with_retry(
                    sys_prompt=JUDGE_SYSTEM_PROMPT,
                    text_prompt=text_prompt,
                    user_content=interleaved,
                    temperature=float(self.frontier.temperature),
                    max_tokens=self.max_tokens,
                    max_retries=self.max_retries,
                    timeout=self._api_timeout,
                )
            except Exception as exc:
                r = self._fail_sample(
                    base=base,
                    status="request_exception",
                    message=f"{type(exc).__name__}: {exc}",
                    dump_stem=dump_stem,
                    edited=edited,
                    extra={"interleaved_redacted": dump_request, "api_summary": None},
                )
                return r, None

            api_summary = _summarize_api_result(result)
            if isinstance(result, dict) and result.get("_error") is not None:
                err_txt = str(result.get("_error"))
                r = self._fail_sample(
                    base=base,
                    status="api_error",
                    message=err_txt,
                    dump_stem=dump_stem,
                    edited=edited,
                    extra={"api_summary": api_summary, "interleaved_redacted": dump_request},
                )
                return r, None

            ok, response_text, err = self.frontier.client.extract_message_from_response(result)
            if not ok or response_text is None:
                msg = err or "empty or unparseable assistant message"
                if parse_try + 1 < TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES:
                    FRONTIER_JUDGE_LOG.warning(
                        "[%s] extract_failed (%s); parse_try %s/%s — retrying judge completion",
                        call_id,
                        msg,
                        parse_try + 1,
                        TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES,
                    )
                    continue
                r = self._fail_sample(
                    base=base,
                    status="extract_failed",
                    message=f"{msg}; api_summary={api_summary!r}",
                    dump_stem=dump_stem,
                    edited=edited,
                    extra={"api_summary": api_summary, "interleaved_redacted": dump_request},
                )
                return r, None

            rt_full = str(response_text)
            rt_stored = _truncate_text(rt_full, self._max_dump_response_chars)
            parsed = try_parse_judge_output(rt_full)
            if parsed is None:
                if parse_try + 1 < TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES:
                    FRONTIER_JUDGE_LOG.warning(
                        "[%s] judge XML/JSON parse failed; parse_try %s/%s — retrying judge completion",
                        call_id,
                        parse_try + 1,
                        TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES,
                    )
                    continue
                if self._unparsed_dump_dir is not None:
                    stem_unparsed = (
                        dump_stem
                        if dump_stem is not None
                        else _make_judge_response_dump_stem(
                            run_id=self._dump_run_id,
                            intra_batch_index=intra_batch_index,
                            trajectory_id=trajectory_id,
                            request_index=request_index,
                        )
                    )
                    write_toolgen_judge_unparsed_response_dump(
                        self._unparsed_dump_dir,
                        stem=stem_unparsed,
                        raw_response=rt_full,
                        meta={
                            "reward_model": "toolgen_searchbetter_judge_frontier",
                            "call_id": call_id,
                            "model_name": self.frontier_model_name,
                            "status": "parse_error",
                            "api_summary": api_summary,
                        },
                    )
                r = self._fail_sample(
                    base=base,
                    status="parse_error",
                    message=(
                        "Failed to parse strict XML/JSON from Frontier response; "
                        f"first 500 chars: {rt_full[:500]!r}"
                    ),
                    dump_stem=dump_stem,
                    edited=edited,
                    extra={
                        "api_summary": api_summary,
                        "response_text": rt_stored,
                        "response_truncated": len(rt_stored) < len(rt_full),
                        "interleaved_redacted": dump_request,
                    },
                )
                return r, None

            try:
                labeled = parsed_judge_labeled_scores_03(parsed)
                reward = float(
                    parsed_judge_json_to_weighted_reward(parsed, rubric, **self._judge_reward_weights)
                )
            except (ValueError, TypeError, KeyError) as exc:
                if parse_try + 1 < TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES:
                    FRONTIER_JUDGE_LOG.warning(
                        "[%s] invalid parsed judge scores (%s: %s); parse_try %s/%s — retrying judge completion",
                        call_id,
                        type(exc).__name__,
                        exc,
                        parse_try + 1,
                        TOOLGEN_JUDGE_PARSE_COMPLETION_TRIES,
                    )
                    continue
                r = self._fail_sample(
                    base=base,
                    status="invalid_parsed_output",
                    message=f"{type(exc).__name__}: {exc}",
                    dump_stem=dump_stem,
                    edited=edited,
                    extra={
                        "api_summary": api_summary,
                        "response_text": rt_stored,
                        "response_truncated": len(rt_stored) < len(rt_full),
                        "interleaved_redacted": dump_request,
                        "parsed": parsed,
                    },
                )
                return r, None
            out = {
                **base,
                "status": "ok",
                "error": None,
                "api_summary": api_summary,
                "response_text": rt_stored,
                "response_truncated": len(rt_stored) < len(rt_full),
                "parsed": parsed,
                "labeled_scores_03": labeled,
                "reward": reward,
            }
            if dump_request is not None:
                out["interleaved_redacted"] = dump_request
            if dump_stem is not None:
                self._persist_eval_artifact_pair(dump_stem, out, edited)

            return reward, labeled

        raise RuntimeError(
            "ToolGenSearchBetterJudgeFrontierRewardModel: exhausted parse completion retries without success"
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Any]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        user_prompt: Optional[List[str]] = None,
        verification_checklist: Optional[List[Any]] = None,
        evaluation_rubric: Optional[List[Any]] = None,
        augmented_generation_details: Optional[List[Any]] = None,
        trajectory_id: Optional[List[Any]] = None,
        request_index: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> RewardModelOutput:
        del kwargs
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided for ToolGenSearchBetterJudgeFrontier")
        if condition_images is None:
            raise ValueError("condition_images is required")
        if user_prompt is None:
            raise ValueError("user_prompt is required")

        batch_len = len(prompt)
        if len(image) != batch_len or len(condition_images) != batch_len:
            raise ValueError(
                f"expected len(prompt)==len(image)==len(condition_images)==batch_size, got "
                f"{len(prompt)}, {len(image)}, {len(condition_images)}"
            )
        validate_meta_list("user_prompt", user_prompt, batch_len)
        for i, up in enumerate(user_prompt):
            if up is None:
                raise ValueError(
                    f"sample {i}: user_prompt is required but got None. "
                    "Store user_prompt on every sample (top-level or extra_kwargs from JSONL)."
                )
        if verification_checklist is None:
            verification_checklist = [[] for _ in range(batch_len)]
        else:
            validate_meta_list("verification_checklist", verification_checklist, batch_len)
        if evaluation_rubric is None:
            evaluation_rubric = [{} for _ in range(batch_len)]
        else:
            validate_meta_list("evaluation_rubric", evaluation_rubric, batch_len)

        if augmented_generation_details is None:
            augmented_generation_details = [None] * batch_len
        else:
            validate_meta_list("augmented_generation_details", augmented_generation_details, batch_len)

        if trajectory_id is None:
            trajectory_id = [""] * batch_len
        else:
            validate_meta_list("trajectory_id", trajectory_id, batch_len)

        if request_index is None:
            request_index = [-1] * batch_len
        else:
            validate_meta_list("request_index", request_index, batch_len)

        ref_lists = [coerce_ref_image_list(c) for c in condition_images]
        meta_checklists = [as_str_list(verification_checklist[i], sample_index=i) for i in range(batch_len)]
        meta_rubrics = [as_rubric_dict(evaluation_rubric[i], sample_index=i) for i in range(batch_len)]

        FRONTIER_JUDGE_LOG.info(
            "Frontier judge batch: %s samples, max_concurrent=%s, api_timeout=%ss, dump=%s, error_policy=%s",
            batch_len,
            self.max_concurrent,
            self._api_timeout,
            str(self._dump_dir) if self._dump_dir else "off",
            self._error_policy,
        )

        work = [
            (
                i,
                str(user_prompt[i]),
                image[i],
                ref_lists[i],
                meta_checklists[i],
                meta_rubrics[i],
                augmented_generation_details[i],
                str(trajectory_id[i] or ""),
                request_index[i],
            )
            for i in range(batch_len)
        ]
        scores: List[Optional[float]] = [None] * batch_len
        labeled_batch: List[Optional[List[tuple[str, float]]]] = [None] * batch_len
        max_workers = max(1, self.max_concurrent)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx: Dict[Any, int] = {}
            for idx, wp, ed, rl, cl, rb, ad, tid, rid in work:
                fut = pool.submit(
                    self._score_one_sync,
                    intra_batch_index=idx,
                    batch_len=batch_len,
                    user_prompt=wp,
                    edited=ed,
                    refs=rl,
                    checklist=cl,
                    rubric=rb,
                    aug_detail=ad,
                    trajectory_id=tid,
                    request_index=rid,
                )
                future_to_idx[fut] = idx
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                r, lab = fut.result()
                scores[idx] = r
                labeled_batch[idx] = lab

        rewards = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return RewardModelOutput(
            rewards=rewards,
            extra_info={"judge_labeled_scores_batch": labeled_batch},
        )
