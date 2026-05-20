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

# src/flow_factory/trainers/grpo.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""
import os
import traceback
from typing import List, Dict, Optional, Any, Union, Literal, Callable
from functools import partial
from collections import defaultdict
import torch
import numpy as np
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import GRPOTrainingArguments
from ..samples import BaseSample, DPO_PREFERENCE_CANDIDATE_FLAG
from ..models.abc import BaseAdapter
from ..rewards import RewardBuffer
from ..advantage import AdvantageProcessor
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import TrajectoryCollector, compute_trajectory_indices
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from ..utils.dist import reduce_loss_info
from diffusers.utils.torch_utils import randn_tensor

logger = setup_logger(__name__)


# ============================ GRPO Trainer ============================
class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    References:
    [1] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args : GRPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps
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
            pgs = int(self.training_args.group_size) + int(self.training_args.preference_extra_candidates)
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

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    @property
    def enable_sft_candidate(self) -> bool:
        return (
            self.training_args.sft_candidate_mode
            and self.training_args.preference_extra_candidates > 0
        )

    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and 
                self.epoch % self.log_args.save_freq == 0 and 
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0 and
                self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self._gather_train_samples_and_maybe_dump(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    # =========================== Evaluation Loop ============================
    def evaluate(self) -> None:
        """Evaluation loop."""
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
                        'trajectory_indices': None, # No need to store trajectories during evaluation
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
                
                flat_start_eval = self._gather_object_flat_start_index(len(all_samples))

                # Log statistics
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

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        self.reward_buffer.clear() # Clear reward buffer
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        pref_extra = self.training_args.preference_extra_candidates
        k = self.training_args.group_size
        if pref_extra > 0:
            sampler_name = self.config.data_args.sampler_type
            if sampler_name != "group_contiguous":
                raise RuntimeError(
                    "train.preference_extra_candidates > 0 requires data.sampler_type='group_contiguous' "
                    f"Got sampler_type={sampler_name!r}."
                )

        rollout_pending: List[BaseSample] = []

        def _finalize_group(chunk: List[BaseSample]) -> List[BaseSample]:
            self._maybe_offload_samples_to_cpu(chunk)
            if pref_extra <= 0:
                return list(chunk)
            return self._inject_preference_candidates(list(chunk))

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices,
                    **batch,
                }
                sample_kwargs = self._materialize_jsonl_images_for_adapter_inference(sample_kwargs)
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)

                if pref_extra > 0:
                    for s in sample_batch:
                        rollout_pending.append(s)
                        if len(rollout_pending) == k:
                            group_with_candidates = _finalize_group(rollout_pending)
                            samples.extend(group_with_candidates)
                            self.reward_buffer.add_samples(group_with_candidates)
                            rollout_pending = []
                else:
                    self._maybe_offload_samples_to_cpu(sample_batch)
                    samples.extend(sample_batch)
                    self.reward_buffer.add_samples(sample_batch)

        if rollout_pending:
            self._maybe_offload_samples_to_cpu(rollout_pending)
            samples.extend(rollout_pending)
            self.reward_buffer.add_samples(rollout_pending)

        return samples

    # =========================== Reward / advantage (Stages 4--5) ============================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards from the buffer, compute advantages, and log advantage metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        # Candidates get advantage=0 so they don't contribute to RL loss if they slip through
        if self.enable_sft_candidate:
            for s in samples:
                if self._is_preference_candidate_sample(s):
                    s.advantage = torch.tensor(0.0)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self._maybe_stamp_train_samples_predicted_dump_png_paths(
                adv_metrics, num_local_rollout_samples=len(samples),
            )
            self.log_data(adv_metrics, step=self.step)

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): PPO-style clipped loss and optional KL."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size

        # Separate SFT candidate samples from RL samples
        sft_candidate_samples: List[BaseSample] = []
        rl_samples: List[BaseSample] = samples
        if self.enable_sft_candidate:
            rl_samples = [s for s in samples if not self._is_preference_candidate_sample(s)]
            sft_candidate_samples = [s for s in samples if self._is_preference_candidate_sample(s)]

        num_batches = (len(rl_samples) + per_device_batch_size - 1) // per_device_batch_size

        # Pre-batch SFT candidates (cycle through them during RL training)
        sft_batches: List[List[BaseSample]] = []
        if sft_candidate_samples:
            sft_batches = [
                sft_candidate_samples[i:i + per_device_batch_size]
                for i in range(0, len(sft_candidate_samples), per_device_batch_size)
            ]

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(rl_samples), generator=perm_gen)
            shuffled_samples = [rl_samples[i] for i in perm]

            self.adapter.train()
            loss_info = defaultdict(list)

            # Lazy per-batch reload: only the current micro-batch lives on GPU.
            # When samples are GPU-resident `sample.to(device)` is a no-op; when
            # they are CPU-resident (offload pipeline) this is the H2D point.
            with self.autocast():
                for batch_idx in tqdm(
                    range(num_batches),
                    total=num_batches,
                    desc=f'Epoch {self.epoch} Training',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    start = batch_idx * per_device_batch_size
                    batch_samples = [
                        sample.to(device)
                        for sample in shuffled_samples[start:start + per_device_batch_size]
                    ]
                    batch = BaseSample.stack(batch_samples)
                    latents_index_map = batch['latent_index_map']  # (T+1,) LongTensor
                    log_probs_index_map = batch['log_prob_index_map']  # (T,) LongTensor
                    # Iterate through timesteps
                    for idx, timestep_index in enumerate(tqdm(
                        self.adapter.scheduler.train_timesteps,
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Prepare inputs
                            # Get old log prob
                            old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                            # Get current timestep data
                            num_timesteps = batch['timesteps'].shape[1]
                            t = batch['timesteps'][:, timestep_index]
                            t_next = (
                                batch['timesteps'][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps
                                else torch.tensor(0, device=self.accelerator.device)
                            )
                            # Get latents
                            latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                            next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]
                            # Prepare forward input
                            forward_inputs = {
                                **self.training_args, # Pass kwargs like `guidance_scale` and `do_classifier_free_guidance`
                                't': t,
                                't_next': t_next,
                                'latents': latents,
                                'next_latents': next_latents,
                                'compute_log_prob': True,
                                'noise_level': self.adapter.scheduler.noise_level,
                                **batch
                            }
                            forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                            # 2. Forward pass
                            if self.enable_kl_loss:
                                if self.training_args.kl_type == 'v-based':
                                    return_kwargs = ['log_prob', 'noise_pred', 'dt']
                                elif self.training_args.kl_type == 'x-based':
                                    return_kwargs = ['log_prob', 'next_latents', 'next_latents_mean', 'dt']
                            else:
                                return_kwargs = ['log_prob', 'dt']

                            forward_inputs['return_kwargs'] = return_kwargs
                            output = self.adapter.forward(**forward_inputs)

                            # 3. Compute loss
                            # Clip advantages
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            # PPO-style clipped loss
                            ratio = torch.exp(output.log_prob - old_log_prob)
                            ratio_clip_range = self.training_args.clip_range

                            unclipped_loss = -adv * ratio
                            clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            loss = policy_loss

                            # 3.5 SFT candidate loss (flow-matching velocity MSE)
                            if sft_batches:
                                sft_batch_idx = batch_idx % len(sft_batches)
                                sft_batch_samples = [s.to(device) for s in sft_batches[sft_batch_idx]]
                                sft_batch_stacked = BaseSample.stack(sft_batch_samples)
                                sft_latents = sft_batch_stacked['all_latents'][:, -1]

                                sft_bs = sft_latents.shape[0]
                                sft_timesteps = TimeSampler.uniform(
                                    batch_size=sft_bs,
                                    num_timesteps=1,
                                    timestep_range=(0.0, 0.99),
                                    time_shift=1.0,
                                    device=device,
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

                                _sft_excluded = {'all_latents', 'timesteps', 'advantage', 'log_probs'}
                                sft_fwd_kwargs = {
                                    **self.training_args,
                                    'compute_log_prob': False,
                                    'return_kwargs': ['noise_pred'],
                                    'noise_level': 0.0,
                                    't': sft_t,
                                    't_next': torch.zeros_like(sft_t),
                                    'latents': noised_sft,
                                    **{k: v for k, v in sft_batch_stacked.items()
                                       if k not in _sft_excluded},
                                }
                                sft_fwd_kwargs = filter_kwargs(self.adapter.forward, **sft_fwd_kwargs)
                                sft_output = self.adapter.forward(**sft_fwd_kwargs)
                                sft_target = sft_noise - sft_latents
                                sft_loss_val = ((sft_output.noise_pred.float() - sft_target.float()) ** 2).mean()
                                loss = loss + self.training_args.sft_candidate_weight * sft_loss_val
                                loss_info['sft_diffusion_loss'].append(sft_loss_val.detach())

                            # 4. Compute KL-div
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs['compute_log_prob'] = False
                                    if self.training_args.kl_type == 'v-based':
                                        # KL in velocity space
                                        ref_forward_inputs['return_kwargs'] = ['noise_pred']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                    elif self.training_args.kl_type == 'x-based':
                                        # KL in latent space
                                        ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)

                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                if self.training_args.kl_type == 'v-based':
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    )
                                elif self.training_args.kl_type == 'x-based':
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    )
                                
                                kl_div = torch.mean(kl_div)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            # 5. Log per-timestep info
                            loss_info['ratio'].append(ratio.detach())
                            loss_info['unclipped_loss'].append(unclipped_loss.detach())
                            loss_info['clipped_loss'].append(clipped_loss.detach())
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # 6. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Communicate and log losses
                                loss_info = reduce_loss_info(self.accelerator, loss_info)
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal['sum', 'gdpo'], Callable]] = None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style), or a custom callable.
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )


# ============================ GRPO-Guard Trainer ============================
class GRPOGuardTrainer(GRPOTrainer):
    """
    GRPOGuard Trainer with reweighted loss.
    References:
    [1] GRPO-Guard: https://arxiv.org/abs/2510.22319
    [2] Temp-FlowGRPO: https://arxiv.org/abs/2508.04324
    """

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices, # Selectively store required trajectory positions for memory efficiency
                    'extra_call_back_kwargs': ['next_latents_mean'], # For GRPO-Guard, we need to store `next_latents_mean` for ratio normalization
                    **batch,
                }
                sample_kwargs = self._materialize_jsonl_images_for_adapter_inference(sample_kwargs)
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                # Deterministic D2H so reward_buffer sees CPU-resident samples
                # (no-op when offload_samples_to_cpu is False).
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): GRPO-Guard reweighted loss and optional KL."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            self.adapter.train()
            loss_info = defaultdict(list)

            # Lazy per-batch reload: only the current micro-batch lives on GPU.
            with self.autocast():
                for batch_idx in tqdm(
                    range(num_batches),
                    total=num_batches,
                    desc=f'Epoch {self.epoch} Training',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    start = batch_idx * per_device_batch_size
                    batch_samples = [
                        sample.to(device)
                        for sample in shuffled_samples[start:start + per_device_batch_size]
                    ]
                    batch = BaseSample.stack(batch_samples)
                    latents_index_map = batch['latent_index_map']  # (T+1,) LongTensor
                    log_probs_index_map = batch['log_prob_index_map']  # (T,) LongTensor
                    callback_index_map = batch['callback_index_map'][0]  # (T,) LongTensor, shared across batch.
                    # Iterate through timesteps
                    for idx, timestep_index in enumerate(tqdm(
                        self.adapter.scheduler.train_timesteps,
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Prepare inputs
                            # Get old log prob
                            old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                            # Get current timestep data
                            num_timesteps = batch['timesteps'].shape[1]
                            t = batch['timesteps'][:, timestep_index]
                            t_next = (
                                batch['timesteps'][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps
                                else torch.tensor(0, device=self.accelerator.device)
                            )
                            # Get latents
                            latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                            next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]
                            # Prepare forward input
                            forward_inputs = {
                                **self.training_args, # Pass kwargs like `guidance_scale` and `do_classifier_free_guidance`
                                't': t,
                                't_next': t_next,
                                'latents': latents,
                                'next_latents': next_latents,
                                'compute_log_prob': True,
                                'noise_level': self.adapter.scheduler.noise_level,
                                **batch
                            }
                            forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                            # 2. Forward pass
                            return_kwargs = set(['log_prob', 'next_latents_mean', 'std_dev_t', 'dt'])
                            if self.enable_kl_loss:
                                if self.training_args.kl_type == 'v-based':
                                    return_kwargs.add('noise_pred')
                                elif self.training_args.kl_type == 'x-based':
                                    return_kwargs.add('next_latents_mean')
                            
                            forward_inputs['return_kwargs'] = list(return_kwargs)
                            output = self.adapter.forward(**forward_inputs)

                            # 3. Compute loss
                            # Clip advantages
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            # Reweighted ratio
                            scale_factor = torch.sqrt(-output.dt) * output.std_dev_t
                            old_next_latents_mean = batch['next_latents_mean'][:, callback_index_map[timestep_index]]
                            mse = (output.next_latents_mean - old_next_latents_mean).flatten(1).pow(2).mean(dim=1)
                            ratio = torch.exp((output.log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))
                            # PPO-style clipped loss
                            ratio_clip_range = self.training_args.clip_range

                            unclipped_loss = -adv * ratio
                            clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            loss = policy_loss

                            # 4. Compute KL-div
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs['compute_log_prob'] = False
                                    if self.training_args.kl_type == 'v-based':
                                        # KL in velocity space
                                        ref_forward_inputs['return_kwargs'] = ['noise_pred']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                    elif self.training_args.kl_type == 'x-based':
                                        # KL in latent space
                                        ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)

                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                if self.training_args.kl_type == 'v-based':
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    )
                                elif self.training_args.kl_type == 'x-based':
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    )

                                kl_div = torch.mean(kl_div)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            # 5. Log per-timestep info
                            loss_info['ratio'].append(ratio.detach())
                            loss_info['unclipped_loss'].append(unclipped_loss.detach())
                            loss_info['clipped_loss'].append(clipped_loss.detach())
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # 6. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Communicate and log losses
                                loss_info = reduce_loss_info(self.accelerator, loss_info)
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)