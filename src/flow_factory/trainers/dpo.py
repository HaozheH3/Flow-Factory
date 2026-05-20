# Copyright 2026 Jayce-Ping
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

# src/flow_factory/trainers/dpo.py
"""
Diffusion-DPO (Direct Preference Optimization) Trainer.
Implements online DPO for flow matching models using velocity MSE (target = noise - x_0).

References:
[1] Diffusion Model Alignment Using Direct Preference Optimization
    - https://arxiv.org/abs/2311.12908
[2] flow_grpo reference implementation
    - https://github.com/yifan123/flow_grpo
"""

import os
import math
import traceback
from collections import defaultdict
from dataclasses import fields as dc_fields
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import broadcast_object_list
import torch.nn.functional as F
import tqdm as tqdm_
from diffusers.utils.torch_utils import randn_tensor

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import DPOTrainingArguments
from ..samples import BaseSample, DPO_PREFERENCE_CANDIDATE_FLAG
from ..rewards.reward_processor import RewardBuffer
from ..advantage.advantage_processor import AdvantageProcessor
from ..models.abc import BaseAdapter
from PIL import Image

from ..data_utils.dataset import _resolve_path
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    to_broadcast_tensor,
)
from ..utils.dist import gather_samples
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from ..utils.image_similarity import max_ssim_against_conditions
from ..rewards.toolgen_searchbetter_judge_common import (
    TOOLGEN_JUDGE_LABELED_SCORES_KEY,
    TOOLGEN_JUDGE_LABELED_SCORES_LOG_CACHE,
    toolgen_dpo_decompose_major_rest_native03,
)
from ..logger.formatting import (
    build_dpo_all_candidate_images_log_images,
    build_dpo_dataset_preference_candidate_images,
    build_dpo_preference_pair_log_images,
)


logger = setup_logger(__name__)

DPO_MAX_CONDITION_SSIM_KEY = "dpo_max_condition_ssim"
DPO_PAIR_SELECTION_SCORE_KEY = "dpo_pair_selection_score"


class DPOTrainer(BaseTrainer):
    """
    Diffusion-DPO Trainer for Flow Matching models.

    Implements online DPO: generates multiple samples per prompt via K-repeat
    sampling, scores them with reward models, forms chosen/rejected pairs from
    the best/worst within each group, then optimises a velocity MSE DPO loss against a frozen reference model.

    Loss:
        L = -log sigma(-beta/2 * ((theta_w_err - ref_w_err) - (theta_l_err - ref_l_err)))
    where err = MSE(noise_pred, noise - x_0) averaged over spatial dims (same as flow_grpo train_sd3_dpo).

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: DPOTrainingArguments
        self.num_train_timesteps = self.training_args.num_train_timesteps
        if self.training_args.preference_extra_candidates > 0:
            if (
                type(self.adapter).dpo_clone_sample_with_preference_image
                is BaseAdapter.dpo_clone_sample_with_preference_image
            ):
                raise TypeError(
                    "train.preference_extra_candidates > 0 requires an adapter that overrides "
                    "dpo_clone_sample_with_preference_image (e.g. Flux2KleinAdapter)."
                )

    def _init_reward_model(self):
        """Use enlarged group size for training reward buffer when injecting preference candidates."""
        super()._init_reward_model()
        if self.training_args.preference_extra_candidates > 0:
            pgs = self.training_args.preference_group_size
            self.reward_buffer = RewardBuffer(self.reward_processor, pgs)
            train_reward_configs = self.reward_loader.get_reward_configs('train')
            self.advantage_processor = AdvantageProcessor(
                accelerator=self.accelerator,
                reward_weights={
                    name: cfg.weight
                    for name, cfg in train_reward_configs.items()
                },
                group_size=pgs,
                global_std=getattr(self.training_args, 'global_std', True),
                sampler_type=self.config.data_args.sampler_type,
                verbose=self.log_args.verbose,
                max_train_samples_for_log=self.log_args.log_max_train_samples,
            )

    def _dpo_inject_preference_candidates(self, sample_batch: List[BaseSample]) -> List[BaseSample]:
        """Append preference-image candidates after a **complete** rollout group.

        ``sample_batch`` must contain ``group_size`` policy rollouts sharing the same
        ``sample.unique_id`` (one dataset row, K stochastic generations). This method emits
        those K samples followed by ``preference_extra_candidates`` packed clones —
        effectively ``group_size + preference_extra_candidates`` rows per dataset row.

        Caller :meth:`sample` gathers K rollouts before invoking this when preference
        extras are enabled and ``sampler_type == "group_contiguous"``.
        """
        n_extra = self.training_args.preference_extra_candidates
        if n_extra <= 0:
            return sample_batch
        key = self.training_args.preference_candidate_metadata_key
        base_dir = self.config.data_args.dataset_dir
        expected_k = int(self.training_args.group_size)
        if len(sample_batch) != expected_k:
            raise ValueError(
                "DPO `_dpo_inject_preference_candidates` expected exactly "
                f"group_size ({expected_k}) rollout samples before preference injection; "
                f"got {len(sample_batch)}. Ensure `sample()` batches one full contiguous group "
                "(see `sampler_type='group_contiguous'`)."
            )
        group_uid = sample_batch[0].unique_id
        if any(s.unique_id != group_uid for s in sample_batch[1:]):
            raise ValueError(
                "DPO preference injection received a rollout list whose members do not "
                "share one `unique_id` (expected one dataset row × group_size generations); "
                f"unique_ids={sorted({s.unique_id for s in sample_batch})!r}"
            )

        order: List[Any] = []
        groups: Dict[Any, List[BaseSample]] = defaultdict(list)
        for s in sample_batch:
            uid = s.unique_id
            if uid not in groups:
                order.append(uid)
            groups[uid].append(s)

        out: List[BaseSample] = []
        for uid in order:
            members = groups[uid]
            template = members[0]
            raw_paths = template.extra_kwargs.get(key)
            if raw_paths is None:
                raise KeyError(
                    f"DPO preference_extra_candidates={n_extra} requires extra_kwargs[{key!r}] on each "
                    f"rollout sample (unique_id={uid})."
                )
            if not isinstance(raw_paths, list):
                raise TypeError(
                    f"{key!r} must be a list of image path strings, got {type(raw_paths).__name__} "
                    f"(unique_id={uid})"
                )
            if len(raw_paths) != n_extra:
                raise ValueError(
                    f"{key!r} must have length {n_extra} (= preference_extra_candidates), "
                    f"got {len(raw_paths)} for unique_id={uid}. "
                    "Variable candidate counts across rows break groupwise reward sizing on one rank."
                )
            for s in members:
                if s.extra_kwargs.get(key) != raw_paths:
                    raise ValueError(
                        f"Inconsistent {key!r} within the same unique_id group ({uid!r}): all K rollouts "
                        "must list the same candidate paths."
                    )
            out.extend(members)
            for rel in raw_paths:
                if not isinstance(rel, str) or not rel.strip():
                    raise ValueError(
                        f"Invalid image path in {key!r}: {rel!r} (unique_id={uid})"
                    )
                path = _resolve_path(base_dir, rel)
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"DPO preference candidate image not found: {path!r} (from {rel!r}, unique_id={uid})"
                    )
                pil = Image.open(path).convert("RGB")
                out.append(
                    self.adapter.dpo_clone_sample_with_preference_image(
                        template, pil, key
                    )
                )
        return out

    # ====================== Main Loop ======================
    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Save checkpoint
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0
                and self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self._gather_train_samples_and_maybe_dump(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # ====================== Evaluation ======================
    def evaluate(self) -> None:
        """Evaluation loop — same pattern as GRPO."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        all_samples: List[BaseSample] = []
        gathered_rewards: Dict[str, np.ndarray] = {}
        eval_status = "ok"
        eval_err_tb: Optional[str] = None

        try:
            with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
                for batch in tqdm(
                    self.test_dataloader,
                    desc='Evaluating',
                    disable=not self.show_progress_bar,
                ):
                    generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                    inference_kwargs = {
                        'compute_log_prob': False,
                        'generator': generator,
                        'trajectory_indices': None,
                        **self.eval_args,
                    }
                    inference_kwargs.update(**batch)
                    inference_kwargs = self._materialize_jsonl_images_for_adapter_inference(inference_kwargs)
                    inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                    samples = self.adapter.inference(**inference_kwargs)
                    all_samples.extend(samples)
                    self.eval_reward_buffer.add_samples(samples)

                rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split='pointwise')

                # Gather and log rewards
                rewards = {
                    key: torch.as_tensor(value).to(self.accelerator.device)
                    for key, value in rewards.items()
                }
                gathered_rewards = {
                    key: self.accelerator.gather(value).cpu().numpy()
                    for key, value in rewards.items()
                }

                eval_ssim_mean: Optional[float] = None
                if self.training_args.dpo_condition_ssim_enable:
                    sum_m, n_i2i, _sh = self._dpo_compute_condition_ssim_batch(all_samples)
                    tss = torch.tensor(
                        [sum_m, float(n_i2i), 0.0],
                        device=self.accelerator.device,
                        dtype=torch.float64,
                    )
                    tss = self.accelerator.reduce(tss, reduction="sum")
                    if tss[1].item() > 0:
                        eval_ssim_mean = tss[0].item() / tss[1].item()

                flat_start_eval = self._gather_object_flat_start_index(len(all_samples))

                if self.accelerator.is_main_process:
                    _log_data = {
                        f'eval/reward_{key}_mean': np.mean(value)
                        for key, value in gathered_rewards.items()
                    }
                    _log_data.update(
                        {
                            f'eval/reward_{key}_std': np.std(value)
                            for key, value in gathered_rewards.items()
                        }
                    )
                    if eval_ssim_mean is not None:
                        _log_data["eval/condition_image_ssim"] = eval_ssim_mean
                    logged_eval = self._eval_samples_for_logger(all_samples)
                    self._stamp_eval_logged_samples_predicted_dump_png_paths_if_enabled(
                        logged_eval,
                        flat_start_index=flat_start_eval,
                        eval_status=eval_status,
                    )
                    _log_data['eval_samples'] = logged_eval
                    self.log_data(_log_data, step=self.step)
                self.accelerator.wait_for_everyone()
        except Exception:
            eval_status = "error"
            eval_err_tb = traceback.format_exc()
            raise
        finally:
            self._gather_eval_samples_and_maybe_dump(
                all_samples,
                gathered_rewards,
                status=eval_status,
                error_traceback=eval_err_tb,
            )

    # ====================== Sampling ======================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts — DPO does NOT need log-probs or full trajectories."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        sampler_name = self.config.data_args.sampler_type
        pref_extra = self.training_args.preference_extra_candidates
        k = self.training_args.group_size
        if pref_extra > 0 and sampler_name != "group_contiguous":
            raise RuntimeError(
                "DPO train.preference_extra_candidates > 0 requires data.sampler_type='group_contiguous' "
                "so each rank observes `group_size` contiguous rollouts for one dataset row, then receives "
                "exactly **one** preference candidate image for that row (group_size + preference_extra_candidates "
                f"scores per row). Got sampler_type={sampler_name!r}."
            )
        rollout_pending: List[BaseSample] = []

        def _finalize_policy_chunk(chunk: List[BaseSample]) -> List[BaseSample]:
            """Return rollout chunk with optional preference row(s); buffer order matches sampler."""
            self._maybe_offload_samples_to_cpu(chunk)
            if pref_extra <= 0:
                return list(chunk)
            return self._dpo_inject_preference_candidates(list(chunk))

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,  # DPO doesn't need log-probs
                    'trajectory_indices': [-1],  # Only keep final latents (clean image)
                    **batch,
                }
                sample_kwargs = self._materialize_jsonl_images_for_adapter_inference(sample_kwargs)
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                if pref_extra <= 0:
                    self._maybe_offload_samples_to_cpu(sample_batch)
                    sample_batch = _finalize_policy_chunk(sample_batch)
                    samples.extend(sample_batch)
                    self.reward_buffer.add_samples(sample_batch)
                    continue

                rollout_pending.extend(sample_batch)
                while len(rollout_pending) >= int(k):
                    chunk = rollout_pending[: int(k)]
                    del rollout_pending[: int(k)]
                    extended = _finalize_policy_chunk(chunk)
                    samples.extend(extended)
                    self.reward_buffer.add_samples(extended)

        if pref_extra > 0 and rollout_pending:
            raise RuntimeError(
                "DPO sampling finished with leftover rollouts — incomplete `group_size` block for "
                f"preference injection (pending {len(rollout_pending)} / expected {int(k)}). "
                "Check num_batches_per_epoch vs group_contiguous tiling."
            )

        return samples

    # ====================== Advantage Computation ======================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        The computed advantages respect the user's ``advantage_aggregation``
        setting (``'sum'`` or ``'gdpo'``).  Call ``self.advantage_processor.pop_advantage_metrics``
        after this when logging training statistics.
        """
        aggregation_func = self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # ====================== Pair Formation ======================

    @staticmethod
    def _get_advantage(sample: BaseSample) -> float:
        """Extract scalar advantage from a sample."""
        adv = sample.extra_kwargs['advantage']
        return adv.item() if hasattr(adv, 'item') else float(adv)

    def _get_aggregated_reward(self, sample: BaseSample) -> float:
        """Weighted sum of per-head rewards (same weighting as :class:`AdvantageProcessor`).

        ``extra_kwargs['rewards']`` is populated after :meth:`prepare_feedback`; this scalar is
        *before* group normalization / GDPO stacking (unlike advantage).
        """
        rew_map = sample.extra_kwargs.get('rewards')
        if rew_map is None or len(rew_map) == 0:
            return float('nan')
        total = 0.0
        for name, val in rew_map.items():
            w = float(self.advantage_processor.reward_weights.get(name, 1.0))
            vv = val.item() if torch.is_tensor(val) else float(val)
            total += w * vv
        return total

    @staticmethod
    def _is_preference_candidate_sample(sample: BaseSample) -> bool:
        return sample.extra_kwargs.get(DPO_PREFERENCE_CANDIDATE_FLAG) is True

    def _form_pairs(
        self,
        samples: List[BaseSample],
    ) -> Tuple[List[Tuple[BaseSample, BaseSample]], Dict[str, Any]]:
        """Form (chosen, rejected) pairs from rule-based selection scores.

        Each sample must have ``extra_kwargs['dpo_pair_selection_score']`` set in
        :meth:`prepare_feedback` (weighted aggregate minus SSIM / worst-major penalties).
        Advantages remain on samples for metrics only; they are **not** used to pick pairs.

        The diffusion DPO loss itself depends only on chosen vs rejected latents, not on advantages.
        """
        # When sft_candidate_mode is active, exclude candidates from pair formation —
        # they will be trained with a separate flow-matching MSE loss instead.
        if self.training_args.sft_candidate_mode:
            samples = [s for s in samples if not self._is_preference_candidate_sample(s)]

        if self.advantage_processor.group_on_same_rank:
            # group_contiguous: all K copies on this rank — form pairs locally
            pairs = self._form_pairs_from_advantages(samples)
            stat_pairs = pairs
        else:
            # distributed_k_repeat: gather full samples across ranks so that
            # every group's K copies are available for pairing.
            gather_field_names = [
                f.name for f in dc_fields(samples[0]) if f.name != '_unique_id'
            ]
            global_samples = gather_samples(
                accelerator=self.accelerator,
                samples=samples,
                field_names=gather_field_names,
                device=self.accelerator.device,
            )

            # Form pairs on global data (every group has all K copies)
            all_pairs = self._form_pairs_from_advantages(global_samples)
            
            # Distribute pairs evenly across ranks
            n_pairs = len(all_pairs)
            world_size = max(1, self.accelerator.num_processes)
            rank = self.accelerator.process_index
            if world_size > 1 and n_pairs < world_size:
                raise RuntimeError(
                    "DPOTrainer (distributed_k_repeat): need at least num_processes "
                    f"chosen/rejected pairs for balanced sharding; got {n_pairs}. "
                    "Increase unique prompts/groups or use sampler_type group_contiguous."
                )

            pairs_sharded = all_pairs[rank::world_size]
            stat_pairs = pairs_sharded
            target = (n_pairs + world_size - 1) // world_size if n_pairs else 0
            if pairs_sharded:
                m = len(pairs_sharded)
                pairs = (pairs_sharded * ((target + m - 1) // m))[:target]
                if m < target:
                    logger.warning(
                        "DPOTrainer: cycled local DPO pair shard to equalize per-rank optimize steps "
                        "(sampler_type distributed_k_repeat; local_pairs(%d), padded_to(%d), "
                        "num_processes(%d), process_index(%d), epoch(%d)). "
                        "Some preference pairs are trained more than once on this rank.",
                        m,
                        target,
                        world_size,
                        rank,
                        self.epoch,
                    )
            else:
                pairs = []

        # DPO-specific keys — globally reduced across all ranks (unpadded pairs only)
        _log_data: Dict[str, Any] = {}
        n = len(stat_pairs)
        if n > 0:
            chosen_advs = np.array([self._get_advantage(p[0]) for p in stat_pairs])
            rejected_advs = np.array([self._get_advantage(p[1]) for p in stat_pairs])
            margins = chosen_advs - rejected_advs
            chosen_reward = np.array([self._get_aggregated_reward(p[0]) for p in stat_pairs])
            rejected_reward = np.array([self._get_aggregated_reward(p[1]) for p in stat_pairs])
            pref_as_chosen = float(
                sum(1 for p in stat_pairs if self._is_preference_candidate_sample(p[0]))
            )
            pref_as_rejected = float(
                sum(1 for p in stat_pairs if self._is_preference_candidate_sample(p[1]))
            )
            local_stats = torch.tensor(
                [
                    float(n),
                    float(chosen_advs.sum()),
                    float(rejected_advs.sum()),
                    float(margins.sum()),
                    float(np.nansum(chosen_reward)),
                    float(np.nansum(rejected_reward)),
                    pref_as_chosen,
                    pref_as_rejected,
                ],
                device=self.accelerator.device,
                dtype=torch.float64,
            )
        else:
            local_stats = torch.zeros(8, device=self.accelerator.device, dtype=torch.float64)

        global_stats = self.accelerator.reduce(local_stats, reduction="sum")
        total_n = global_stats[0].item()
        _log_data['train/dpo_num_pairs'] = int(total_n)
        if total_n > 0:
            _log_data['train/dpo_chosen_adv_mean'] = global_stats[1].item() / total_n
            _log_data['train/dpo_rejected_adv_mean'] = global_stats[2].item() / total_n
            _log_data['train/dpo_adv_margin_mean'] = global_stats[3].item() / total_n
            # Aggregate reward *before* advantage normalization (sum_k w_k r_k across reward heads).
            chosen_r_mean = global_stats[4].item() / total_n
            rejected_r_mean = global_stats[5].item() / total_n
            _log_data['train/dpo_chosen_reward_mean'] = chosen_r_mean
            _log_data['train/dpo_rejected_reward_mean'] = rejected_r_mean
            _log_data['train/dpo_pair_reward_chosen_mean'] = chosen_r_mean
            _log_data['train/dpo_pair_reward_rejected_mean'] = rejected_r_mean
            _log_data['train/dpo_pair_reward_gap_mean'] = chosen_r_mean - rejected_r_mean
            _log_data['train/dpo_pref_candidate_is_chosen_rate'] = global_stats[6].item() / total_n
            _log_data['train/dpo_pref_candidate_is_rejected_rate'] = global_stats[7].item() / total_n

        return pairs, _log_data

    def _dpo_preference_candidate_reward_stats(self, samples: List[BaseSample]) -> Dict[str, Any]:
        """Global mean weighted reward for injected preference candidates (pre-advantage).

        Uses ``accelerator.reduce`` over finite per-sample aggregates. Logs
        ``train/dpo_preference_candidate_reward_mean`` when the global finite count > 0, and always
        ``train/dpo_preference_candidate_count`` (total candidates with a finite weighted reward).
        """
        if self.training_args.preference_extra_candidates <= 0:
            return {}
        vals: List[float] = []
        for s in samples:
            if self._is_preference_candidate_sample(s):
                r = float(self._get_aggregated_reward(s))
                if math.isfinite(r):
                    vals.append(r)
        device = self.accelerator.device
        ssum = float(sum(vals))
        cnt = float(len(vals))
        red = self.accelerator.reduce(
            torch.tensor([ssum, cnt], device=device, dtype=torch.float64),
            reduction="sum",
        )
        global_cnt = red[1].item()
        out: Dict[str, Any] = {}
        if global_cnt > 0.0:
            out["train/dpo_preference_candidate_reward_mean"] = red[0].item() / global_cnt
        out["train/dpo_preference_candidate_count"] = int(global_cnt)
        return out

    def _get_dpo_pair_selection_scalar(self, sample: BaseSample) -> float:
        """Scalar for argmax/argmin within each ``unique_id`` group (rule-modified aggregate)."""
        v = sample.extra_kwargs.get(DPO_PAIR_SELECTION_SCORE_KEY)
        if v is None:
            raise KeyError(
                f"missing {DPO_PAIR_SELECTION_SCORE_KEY!r} for unique_id={sample.unique_id}; "
                "ensure prepare_feedback ran and computed rule-based pair selection scores."
            )
        return float(v.item()) if torch.is_tensor(v) else float(v)

    def _dpo_compute_rule_based_pair_selection_scores(
        self, samples: List[BaseSample],
    ) -> Tuple[int, int, int]:
        """Set ``dpo_pair_selection_score`` = weighted aggregate minus rule penalties (no clamping).

        Returns:
            (count SSIM penalties, count major-worst penalties, count samples scored).
        """
        pen = float(self.training_args.dpo_selection_rule_penalty)
        ssim_thr = float(self.training_args.dpo_penalize_condition_ssim)
        ssim_on = bool(self.training_args.dpo_condition_ssim_enable)

        n_ssim_pen = 0
        n_maj_pen = 0
        n_scored = 0

        uid_to_idxs: Dict[Any, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            uid_to_idxs[s.unique_id].append(i)

        for _uid, idxs in uid_to_idxs.items():
            major_pairs: List[Tuple[int, float]] = []
            for i in idxs:
                vec = samples[i].extra_kwargs.get(TOOLGEN_JUDGE_LABELED_SCORES_KEY)
                if vec is None:
                    vec = samples[i].extra_kwargs.get(TOOLGEN_JUDGE_LABELED_SCORES_LOG_CACHE)
                if vec is not None:
                    mj, _ = toolgen_dpo_decompose_major_rest_native03(list(vec))
                    major_pairs.append((i, float(mj)))

            worst: set[int] = set()
            if major_pairs:
                min_m = min(m for _, m in major_pairs)
                for i, m in major_pairs:
                    if math.isclose(m, min_m, rel_tol=0.0, abs_tol=1e-5):
                        worst.add(i)

            for i in idxs:
                n_scored += 1
                base = self._get_aggregated_reward(samples[i])
                if not math.isfinite(base):
                    raise ValueError(
                        f"sample unique_id={samples[i].unique_id}: non-finite weighted aggregate reward "
                        f"before pair selection ({base!r})"
                    )
                sel = float(base)
                if ssim_on:
                    mx = samples[i].extra_kwargs.get(DPO_MAX_CONDITION_SSIM_KEY)
                    if mx is None:
                        raise KeyError(
                            f"missing {DPO_MAX_CONDITION_SSIM_KEY!r} for unique_id={samples[i].unique_id} "
                            "when dpo_condition_ssim_enable=True; ensure SSIM batch ran in prepare_feedback."
                        )
                    mx_f = float(mx)
                    if mx_f >= 0.0 and mx_f > ssim_thr:
                        sel -= pen
                        n_ssim_pen += 1
                if i in worst:
                    sel -= pen
                    n_maj_pen += 1
                samples[i].extra_kwargs[DPO_PAIR_SELECTION_SCORE_KEY] = sel

        return n_ssim_pen, n_maj_pen, n_scored

    def _form_pairs_from_advantages(
        self,
        samples: List[BaseSample],
    ) -> List[Tuple[BaseSample, BaseSample]]:
        """Form (chosen, rejected) using ``dpo_pair_selection_score`` (rule-modified aggregate)."""
        unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
        _, group_indices = np.unique(unique_ids, return_inverse=True)

        scores = np.array(
            [self._get_dpo_pair_selection_scalar(s) for s in samples],
            dtype=np.float64,
        )

        pairs: List[Tuple[BaseSample, BaseSample]] = []
        for gid in np.unique(group_indices):
            mask = np.where(group_indices == gid)[0]
            if len(mask) < 2:
                logger.warning(f"Group {gid} has less than 2 samples, skipping pair formation.")
                continue
            idxs = mask.astype(np.int64)
            sel = scores[idxs]
            bi = int(np.argmax(sel))
            best = int(idxs[bi])
            others_pos = [j for j in range(len(idxs)) if int(idxs[j]) != best]
            if not others_pos:
                logger.warning(
                    "DPO group %s: only one distinct index after chosen selection; skipping pair.",
                    gid,
                )
                continue
            wi = int(others_pos[int(np.argmin(sel[others_pos]))])
            worst = int(idxs[wi])
            pairs.append((samples[best], samples[worst]))
        return pairs

    def _align_dpo_pairs_across_ranks(
        self,
        pairs: List[Tuple[BaseSample, BaseSample]],
    ) -> List[Tuple[BaseSample, BaseSample]]:
        """Pad local pairs so every rank runs the same number of optimize steps (DDP)."""
        ws = self.accelerator.num_processes
        if ws <= 1 or not dist.is_available() or not dist.is_initialized():
            return pairs

        device = self.accelerator.device
        cnt_t = torch.tensor([len(pairs)], device=device, dtype=torch.long)
        gathered = [torch.zeros_like(cnt_t) for _ in range(ws)]
        dist.all_gather(gathered, cnt_t)
        counts = [int(x.item()) for x in gathered]
        max_cnt = max(counts)
        if max_cnt == 0:
            return pairs

        src = min(i for i, c in enumerate(counts) if c > 0)
        template: Optional[Tuple[BaseSample, BaseSample]] = None
        if min(counts) == 0:
            obj_list = [pairs[0] if pairs else None]
            broadcast_object_list(obj_list, from_process=src)
            template = obj_list[0]
            if template is None:
                raise RuntimeError(
                    "DPOTrainer: cross-rank broadcast of a template preference pair returned None. "
                    f"Expected rank {src} (first rank with local pairs) to broadcast a non-empty pair "
                    "when some ranks have zero pairs; check pair formation and sampler alignment."
                )

        if not pairs:
            if template is None:
                raise RuntimeError(
                    "DPOTrainer: this rank has no DPO pairs but no template pair is available to pad "
                    "to max_pairs_per_rank across ranks. This should not happen after a successful "
                    "broadcast when min(counts)==0; check distributed state and pair formation."
                )
            logger.warning(
                "DPOTrainer: no local pairs on this rank; filled with broadcast template pairs to "
                "match max_pairs_per_rank(%d) across ranks (num_processes(%d), process_index(%d), "
                "epoch(%d)). Training repeats the same preference pair; prefer sampler_type "
                "group_contiguous or more groups per epoch if this persists.",
                max_cnt,
                ws,
                self.accelerator.process_index,
                self.epoch,
            )
            pairs = [template] * max_cnt
        elif len(pairs) < max_cnt:
            n_before = len(pairs)
            out = list(pairs)
            k = 0
            base_len = len(pairs)
            while len(out) < max_cnt:
                out.append(pairs[k % base_len])
                k += 1
            pairs = out
            logger.warning(
                "DPOTrainer: cycled local pairs to match max_pairs_per_rank(%d) across ranks "
                "(local_pairs(%d), padded_to(%d), num_processes(%d), process_index(%d), epoch(%d)). "
                "Some preference pairs receive extra gradient steps on this rank.",
                max_cnt,
                n_before,
                max_cnt,
                ws,
                self.accelerator.process_index,
                self.epoch,
            )
        return pairs

    # ====================== Timestep Sampling ======================
    def _sample_timesteps(self, batch_size: int, num_timesteps: int, timestep_range: Tuple[float, float]) -> torch.Tensor:
        """Sample T×B timesteps for DPO training.

        Reuses ``TimeSampler`` from ``utils.noise_schedule``.
        Rescales output to ``timestep_range`` configured on the training args.

        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with values
            in [t_lo, t_hi].
        """
        device = self.accelerator.device
        if self.training_args.weighting_scheme == 'logit_normal':
            t = TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=num_timesteps,
                timestep_range=timestep_range,
                logit_mean=self.training_args.logit_mean,
                logit_std=self.training_args.logit_std,
                time_shift=self.training_args.time_shift,
                device=device,
                stratified=False,
            )  # (T, B)
        else:  # uniform
            t = TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=num_timesteps,
                timestep_range=timestep_range,
                time_shift=self.training_args.time_shift,
                device=device,
            )
        return t

    # ====================== Forward Helpers ======================
    def _forward_noise_pred(self, latents: torch.Tensor, base_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Run a single forward pass and return the noise prediction."""
        fwd_kwargs = {**base_kwargs, 'latents': latents}
        fwd_kwargs = filter_kwargs(self.adapter.forward, **fwd_kwargs)
        return self.adapter.forward(**fwd_kwargs).noise_pred

    # ====================== Reward / advantage (Stages 4--5) ======================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards, compute advantages, and log advantage-processor metrics.

        Does not form chosen/rejected pairs; :meth:`optimize` calls :meth:`_form_pairs` after
        advantages are stored on each sample.
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        if self.training_args.dpo_condition_ssim_enable:
            sum_m, n_i2i, sum_high = self._dpo_compute_condition_ssim_batch(samples)
            t = torch.tensor(
                [sum_m, float(n_i2i), sum_high],
                device=self.accelerator.device,
                dtype=torch.float64,
            )
            t = self.accelerator.reduce(t, reduction="sum")
            if self.accelerator.is_main_process and t[1].item() > 0:
                self.log_data(
                    {
                        "train/condition_image_ssim": t[0].item() / t[1].item(),
                        "train/high_ssim_rate": t[2].item() / t[1].item(),
                    },
                    step=self.step,
                )
        self.compute_advantages(samples, rewards, store_to_samples=True)
        n_ssim_pen, n_maj_pen, n_s = self._dpo_compute_rule_based_pair_selection_scores(samples)
        tb = torch.tensor(
            [float(n_ssim_pen), float(n_maj_pen), float(n_s)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        tb = self.accelerator.reduce(tb, reduction="sum")
        if self.accelerator.is_main_process and tb[2].item() > 0:
            denom = tb[2].item()
            self.log_data(
                {
                    "train/dpo_selection_ssim_penalty_rate": tb[0].item() / denom,
                    "train/dpo_selection_major_worst_penalty_rate": tb[1].item() / denom,
                },
                step=self.step,
            )
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self._maybe_stamp_train_samples_predicted_dump_png_paths(
                adv_metrics, num_local_rollout_samples=len(samples),
            )
            self.log_data(adv_metrics, step=self.step)

    def _dpo_compute_condition_ssim_batch(
        self, samples: List[BaseSample],
    ) -> Tuple[float, int, float]:
        """Returns local (sum max SSIM, n_i2i with conditions, count high-ssim)."""
        sum_m = 0.0
        n_i2i = 0
        sum_high = 0.0
        thr_hi = self.training_args.dpo_high_ssim_log_threshold
        for s in samples:
            cond = getattr(s, "condition_images", None)
            if cond is None or (isinstance(cond, (list, tuple)) and len(cond) == 0):
                s.extra_kwargs[DPO_MAX_CONDITION_SSIM_KEY] = -1.0
                continue
            if s.image is None:
                raise ValueError(
                    f"sample unique_id={s.unique_id} has condition_images but image is None; "
                    "cannot compute condition SSIM."
                )
            mx, _ = max_ssim_against_conditions(s.image, cond)
            s.extra_kwargs[DPO_MAX_CONDITION_SSIM_KEY] = float(mx)
            sum_m += float(mx)
            n_i2i += 1
            if float(mx) > thr_hi:
                sum_high += 1.0
        return sum_m, n_i2i, sum_high

    # ====================== Optimization ======================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): build chosen/rejected pairs, then DPO preference loss.

        Requires :meth:`prepare_feedback` in the same epoch so ``extra_kwargs['advantage']`` is set
        (for metrics) and ``dpo_pair_selection_score`` is set (for chosen/rejected).
        """
        pairs, pair_log_data = self._form_pairs(samples)
        cand_stats = self._dpo_preference_candidate_reward_stats(samples)
        merged = {**pair_log_data, **cand_stats}
        self.log_data(merged, step=self.step)

        if (
            self.training_args.dpo_log_pair_media_enable
            and self.training_args.dpo_log_pair_media_every_epochs > 0
            and self.epoch % self.training_args.dpo_log_pair_media_every_epochs == 0
            and self.accelerator.is_main_process
            and self.logger is not None
        ):
            backend = self.config.log_args.logging_backend
            if backend not in (None, "none"):
                strips = build_dpo_preference_pair_log_images(
                    pairs,
                    self.training_args.dpo_log_pair_media_max_pairs,
                    reward_weights=dict(self.advantage_processor.reward_weights),
                )
                if strips:
                    self.log_data({"train/dpo_preference_pairs": strips}, step=self.step)

        if (
            self.training_args.dpo_log_preference_candidate_media_enable
            and self.training_args.dpo_log_pair_media_every_epochs > 0
            and self.epoch % self.training_args.dpo_log_pair_media_every_epochs == 0
            and self.accelerator.is_main_process
            and self.logger is not None
        ):
            backend = self.config.log_args.logging_backend
            if backend not in (None, "none"):
                strips_all = build_dpo_all_candidate_images_log_images(
                    samples,
                    self.training_args.dpo_log_pair_media_max_pairs,
                    reward_weights=dict(self.advantage_processor.reward_weights),
                    include_condition_images_in_strip=(
                        self.training_args.dpo_log_preference_candidate_media_include_conditions
                    ),
                    trim_to_candidate_count=None,
                )
                strips_ds = build_dpo_dataset_preference_candidate_images(
                    samples,
                    self.training_args.dpo_log_pair_media_max_pairs,
                    dataset_dir=str(self.config.data_args.dataset_dir),
                    candidate_metadata_key=self.training_args.preference_candidate_metadata_key,
                    include_condition_images_in_strip=(
                        self.training_args.dpo_log_preference_candidate_media_include_conditions
                    ),
                )
                to_log: Dict[str, Any] = {}
                if strips_all:
                    to_log["train/dpo_all_candidate_images"] = strips_all
                if strips_ds:
                    to_log["train/dpo_dataset_preference_candidate_images"] = strips_ds
                if to_log:
                    self.log_data(to_log, step=self.step)

        global_pair_count = int(pair_log_data.get("train/dpo_num_pairs", 0))
        if global_pair_count == 0:
            raise RuntimeError(
                f"DPOTrainer: no valid chosen/rejected pairs at epoch {self.epoch}. "
                "Each prompt group needs at least two samples with comparable advantages to form "
                "a winner and a loser. Check group_size, reward models, and advantage_aggregation."
            )

        pairs = self._align_dpo_pairs_across_ranks(pairs)

        # Collect SFT candidate samples (used only when sft_candidate_mode is active)
        sft_candidate_samples: List[BaseSample] = []
        if self.training_args.sft_candidate_mode:
            sft_candidate_samples = [
                s for s in samples if self._is_preference_candidate_sample(s)
            ]

        # Optimize
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle pairs
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(pairs), generator=perm_gen)
            shuffled_pairs = [pairs[i] for i in perm]

            # Batch pairs
            batch_size = self.training_args.per_device_batch_size
            pair_batches = [
                shuffled_pairs[i:i + batch_size]
                for i in range(0, len(shuffled_pairs), batch_size)
            ]

            # Batch SFT candidates (cycle to match number of DPO pair batches)
            sft_batches: List[List[BaseSample]] = []
            if sft_candidate_samples:
                _sft_flat = list(sft_candidate_samples)
                sft_batches = [
                    _sft_flat[i:i + batch_size]
                    for i in range(0, len(_sft_flat), batch_size)
                ]

            self.adapter.train()
            loss_info = defaultdict(list)

            with self.autocast():
                for pair_batch_idx, pair_batch in enumerate(tqdm(
                    pair_batches,
                    total=len(pair_batches),
                    desc=f'Epoch {self.epoch} DPO Training',
                    position=0,
                    disable=not self.show_progress_bar,
                )):
                    # Stack chosen and rejected latents (shared across timesteps).
                    # Lazy reload: when samples are GPU-resident `sample.to(device)` is
                    # a no-op; when they are CPU-resident (offload pipeline) this is
                    # the H2D point.
                    device = self.accelerator.device
                    chosen_samples = [p[0].to(device) for p in pair_batch]
                    rejected_samples = [p[1].to(device) for p in pair_batch]

                    chosen_batch = BaseSample.stack(chosen_samples)
                    rejected_batch = BaseSample.stack(rejected_samples)

                    # Get clean latents (final step from trajectory, index -1)
                    chosen_latents = chosen_batch['all_latents'][:, -1]
                    rejected_latents = rejected_batch['all_latents'][:, -1]

                    current_batch_size = chosen_latents.shape[0]

                    # Pre-sample T×B timesteps for this pair batch
                    all_timesteps = self._sample_timesteps(
                        batch_size=current_batch_size,
                        num_timesteps=self.num_train_timesteps,
                        timestep_range=self.training_args.timestep_range,
                    )  # (T, B)

                    # Build static forward kwargs (shared across timesteps)
                    _excluded_batch_keys = {'all_latents', 'timesteps', 'advantage'}
                    static_kwargs = {
                        **self.training_args,
                        'compute_log_prob': False,
                        'return_kwargs': ['noise_pred'],
                        'noise_level': 0.0,
                        **{k: v for k, v in chosen_batch.items()
                           if k not in _excluded_batch_keys},
                    }

                    for t_idx in range(self.num_train_timesteps):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            t = all_timesteps[t_idx]  # (B,), scheduler scale [0, 1000]
                            sigma = flow_match_sigma(t)  # σ ∈ [0, 1]
                            noise = randn_tensor(
                                chosen_latents.shape,
                                device=chosen_latents.device,
                                dtype=chosen_latents.dtype,
                            )

                            sigma_broadcast = to_broadcast_tensor(sigma, chosen_latents)

                            # Noise both at same σ: x_t = (1 - σ) * x_0 + σ * noise
                            noised_chosen = (1 - sigma_broadcast) * chosen_latents + sigma_broadcast * noise
                            noised_rejected = (1 - sigma_broadcast) * rejected_latents + sigma_broadcast * noise

                            # Per-timestep forward kwargs (adapter expects scheduler scale)
                            base_kwargs = {
                                **static_kwargs,
                                't': t,
                                't_next': torch.zeros_like(t),
                            }

                            # Policy forward
                            theta_w_pred = self._forward_noise_pred(noised_chosen, base_kwargs)
                            theta_l_pred = self._forward_noise_pred(noised_rejected, base_kwargs)

                            # Reference forward (frozen)
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_w_pred = self._forward_noise_pred(noised_chosen, base_kwargs)
                                ref_l_pred = self._forward_noise_pred(noised_rejected, base_kwargs)

                            # MSE errors per sample — target is flow-matching velocity (noise - x_0), same as
                            # flow_grpo train_sd3_dpo.py: target = noise - model_input
                            target_w = noise - chosen_latents
                            target_l = noise - rejected_latents
                            spatial_dims = tuple(range(1, theta_w_pred.ndim))
                            theta_w_err = ((theta_w_pred.float() - target_w.float()) ** 2).mean(dim=spatial_dims)
                            theta_l_err = ((theta_l_pred.float() - target_l.float()) ** 2).mean(dim=spatial_dims)
                            ref_w_err = ((ref_w_pred.float() - target_w.float()) ** 2).mean(dim=spatial_dims)
                            ref_l_err = ((ref_l_pred.float() - target_l.float()) ** 2).mean(dim=spatial_dims)

                            # DPO loss
                            beta = self.training_args.beta
                            w_diff = theta_w_err - ref_w_err
                            l_diff = theta_l_err - ref_l_err
                            w_l_diff = w_diff - l_diff
                            inside_term = -0.5 * beta * w_l_diff
                            dpo_loss = -F.logsigmoid(inside_term).mean()

                            # SFT candidate loss (flow-matching velocity MSE)
                            if sft_batches:
                                sft_batch_idx = pair_batch_idx % len(sft_batches)
                                sft_batch_samples = [s.to(device) for s in sft_batches[sft_batch_idx]]
                                sft_batch_stacked = BaseSample.stack(sft_batch_samples)
                                sft_latents = sft_batch_stacked['all_latents'][:, -1]

                                sft_bs = sft_latents.shape[0]
                                sft_timesteps = self._sample_timesteps(
                                    batch_size=sft_bs,
                                    num_timesteps=1,
                                    timestep_range=self.training_args.timestep_range,
                                )
                                sft_t = sft_timesteps[0]
                                sft_sigma = flow_match_sigma(sft_t)
                                sft_noise = randn_tensor(
                                    sft_latents.shape,
                                    device=sft_latents.device,
                                    dtype=sft_latents.dtype,
                                )
                                sft_sigma_bc = to_broadcast_tensor(sft_sigma, sft_latents)
                                noised_sft = (1 - sft_sigma_bc) * sft_latents + sft_sigma_bc * sft_noise

                                _sft_excluded = {'all_latents', 'timesteps', 'advantage'}
                                sft_static_kwargs = {
                                    **self.training_args,
                                    'compute_log_prob': False,
                                    'return_kwargs': ['noise_pred'],
                                    'noise_level': 0.0,
                                    **{k: v for k, v in sft_batch_stacked.items()
                                       if k not in _sft_excluded},
                                }
                                sft_fwd_kwargs = {
                                    **sft_static_kwargs,
                                    't': sft_t,
                                    't_next': torch.zeros_like(sft_t),
                                }
                                sft_pred = self._forward_noise_pred(noised_sft, sft_fwd_kwargs)
                                sft_target = sft_noise - sft_latents
                                sft_loss_val = ((sft_pred.float() - sft_target.float()) ** 2).mean()
                                total_loss = dpo_loss + self.training_args.sft_candidate_weight * sft_loss_val
                            else:
                                total_loss = dpo_loss

                            # Logging metrics
                            with torch.no_grad():
                                implicit_reward_chosen = -0.5 * beta * w_diff
                                implicit_reward_rejected = -0.5 * beta * l_diff
                                implicit_reward_gap = (implicit_reward_chosen - implicit_reward_rejected).mean()
                                implicit_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()

                            loss_info['loss'].append(total_loss.detach())
                            if sft_batches:
                                loss_info['dpo_loss'].append(dpo_loss.detach())
                                loss_info['sft_diffusion_loss'].append(sft_loss_val.detach())
                            loss_info['theta_w_err'].append(theta_w_err.mean().detach())
                            loss_info['theta_l_err'].append(theta_l_err.mean().detach())
                            loss_info['ref_w_err'].append(ref_w_err.mean().detach())
                            loss_info['ref_l_err'].append(ref_l_err.mean().detach())
                            loss_info['implicit_accuracy'].append(implicit_accuracy.detach())
                            loss_info['implicit_reward_chosen'].append(implicit_reward_chosen.mean().detach())
                            loss_info['implicit_reward_rejected'].append(implicit_reward_rejected.mean().detach())
                            loss_info['implicit_reward_gap'].append(implicit_reward_gap.detach())

                            # Backward + optimizer step
                            self.accelerator.backward(total_loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                loss_info = reduce_loss_info(self.accelerator, loss_info)
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
