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

# src/flow_factory/trainers/nft.py
"""
DiffusionNFT Trainer.
Reference: 
[1] DiffusionNFT: Online Diffusion Reinforcement with Forward Process
    - https://arxiv.org/abs/2509.16117
"""
import os
from typing import List, Dict, Any, Union, Optional, Tuple
from functools import partial
from collections import defaultdict
from contextlib import nullcontext, contextmanager
import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import NFTTrainingArguments
from ..samples import BaseSample
from ..rewards import RewardBuffer
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from ..utils.dist import reduce_loss_info

logger = setup_logger(__name__)


def compute_nft_query_group_loss_keep_mask_global(
    aggregated_rewards: np.ndarray,
    gathered_ids: np.ndarray,
    *,
    high_reward_threshold: float,
    low_std_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a global per-sample keep mask (1 = keep loss, 0 = mask) from reward vectors and group ids.

    A group is masked when ``mean(reward in group) > high_reward_threshold`` and
    ``std(reward in group) < low_std_threshold``. If every sample would be masked, the mask is
    reset to all ones so optimization does not degenerate.

    Returns:
        ``(keep_mask_global, stats)`` where ``keep_mask_global`` has the same shape as
        ``aggregated_rewards``, and ``stats`` contains counts for logging.
    """
    if aggregated_rewards.shape != gathered_ids.shape:
        raise ValueError(
            f"aggregated_rewards shape {aggregated_rewards.shape} must match gathered_ids "
            f"{gathered_ids.shape}"
        )
    group_keys, group_indices = np.unique(gathered_ids, return_inverse=True)
    keep_mask_global = np.ones_like(aggregated_rewards, dtype=np.float32)
    filtered_group_count = 0
    num_groups = int(len(group_keys))
    for group_id in range(num_groups):
        group_mask = group_indices == group_id
        group_rewards = aggregated_rewards[group_mask]
        group_mean = float(np.mean(group_rewards))
        group_std = float(np.std(group_rewards))
        if group_mean > high_reward_threshold and group_std < low_std_threshold:
            keep_mask_global[group_mask] = 0.0
            filtered_group_count += 1

    total_samples = int(keep_mask_global.shape[0])
    filtered_samples = int((keep_mask_global == 0.0).sum())
    total_groups = num_groups
    reset_all = False
    if filtered_samples == total_samples:
        keep_mask_global[:] = 1.0
        filtered_samples = 0
        filtered_group_count = 0
        reset_all = True

    stats: Dict[str, Any] = {
        "filtered_samples": filtered_samples,
        "total_samples": total_samples,
        "filtered_groups": filtered_group_count,
        "total_groups": total_groups,
        "reset_all": reset_all,
    }
    return keep_mask_global, stats


class DiffusionNFTTrainer(BaseTrainer):
    """
    DiffusionNFT Trainer with off-policy and continuous timestep support.

    Optional **query-group loss mask** (``nft_query_group_loss_mask`` in :class:`~flow_factory.hparams.NFTTrainingArguments`):
    skip NFT (and KL) loss on samples whose ``unique_id`` group has very high mean reward and very
    low cross-sample reward variance, after aggregating multi-reward outputs with model weights and
    gathering across distributed ranks.

    Reference: https://arxiv.org/abs/2509.16117
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # NFT-specific config (from NFTTrainingArguments)
        self.training_args : NFTTrainingArguments
        self.nft_beta = self.training_args.nft_beta
        self.off_policy = self.training_args.off_policy

        # Timestep sampling config
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range

        self.kl_type = self.training_args.kl_type

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0
    
    @contextmanager
    def sampling_context(self):
        """Context manager for sampling with or without EMA parameters."""
        if self.off_policy:
            with self.adapter.use_ema_parameters():
                yield
        else:
            yield

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample continuous or discrete timesteps based on configured `time_sampling_strategy`.

        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with scheduler-scale ``t`` in ``[0, 1000]``.
        """
        device = self.accelerator.device
        time_sampling_strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']

        if time_sampling_strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif time_sampling_strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        elif time_sampling_strategy.startswith('discrete'):
            discrete_config = {
                'discrete': (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init': (False, False),
            }
            if time_sampling_strategy not in discrete_config:
                raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")

            include_init, force_init = discrete_config[time_sampling_strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")

    # =========================== Evaluation Loop ============================
    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples : List[BaseSample] = []

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
            rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy()
                for key, value in rewards.items()
            }

            # Log statistics
            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
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

            # Sampling: use EMA if off_policy
            with self.sampling_context():
                samples = self.sample()

            self.prepare_feedback(samples)
            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DiffusionNFT."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,
                    'trajectory_indices': [-1], # For NFT, only keep the final latents
                    **batch
                }
                sample_kwargs = self._materialize_jsonl_images_for_adapter_inference(sample_kwargs)
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)


        return samples

    # =========================== Optimization Loop ============================
    def _compute_nft_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NFT forward pass for a single timestep.
        
        Args:
            batch: Batch containing prompt embeddings and other inputs.
            timestep: Timestep tensor of shape (B,) in scheduler scale ``[0, 1000]``.
            noised_latents: Interpolated latents ``x_t = (1-σ) x_1 + σ noise`` with ``σ = t/1000``.
        
        Returns:
            Dict with noise_pred.
        """
        t_b = timestep.view(-1) # Scale [0, 1000]

        forward_kwargs = {
            **self.training_args,
            't': t_b,
            't_next': torch.zeros_like(t_b),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': 0.0,
            **{k: v for k, v in batch.items() if k not in ['all_latents', 'timesteps', 'advantage']},
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        
        output = self.adapter.forward(**forward_kwargs)
        
        return {
            'noise_pred': output.noise_pred,
        }

    @staticmethod
    def _scalar_reward_from_sample_rewards(sample: BaseSample, reward_name: str) -> float:
        extra = sample.extra_kwargs.get("rewards")
        if not isinstance(extra, dict):
            raise TypeError(
                f"expected sample.extra_kwargs['rewards'] to be dict before loss mask, "
                f"got {type(extra).__name__} for reward {reward_name!r}"
            )
        if reward_name not in extra:
            raise KeyError(
                f"sample.extra_kwargs['rewards'] missing key {reward_name!r}; "
                f"keys present: {sorted(extra.keys())}"
            )
        v = extra[reward_name]
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().cpu().item())
        return float(v)

    def _build_query_group_loss_keep_mask(self, samples: List[BaseSample]) -> torch.Tensor:
        """
        Per-sample keep mask (1 = train, 0 = skip loss) from query-group reward statistics.

        A query group (``unique_id``) is masked when, on aggregated rewards (weighted sum across
        reward models, gathered across ranks), ``mean > high_threshold`` and ``std < low_threshold``.
        """
        if not samples:
            raise ValueError("_build_query_group_loss_keep_mask: empty samples")

        reward_names = sorted(self.reward_models.keys())
        if not reward_names:
            return torch.ones(
                len(samples),
                dtype=torch.float32,
                device=self.accelerator.device,
            )

        reward_tensors: Dict[str, torch.Tensor] = {}
        for name in reward_names:
            vals = [self._scalar_reward_from_sample_rewards(s, name) for s in samples]
            reward_tensors[name] = torch.tensor(
                vals, dtype=torch.float32, device=self.accelerator.device
            )

        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in reward_tensors.items()
        }
        aggregated_rewards = np.zeros_like(
            next(iter(gathered_rewards.values())),
            dtype=np.float64,
        )
        for key, reward_array in gathered_rewards.items():
            weight = float(self.reward_models[key].config.weight)
            aggregated_rewards += reward_array * weight

        unique_ids = torch.tensor(
            [int(s.unique_id) for s in samples],
            dtype=torch.int64,
            device=self.accelerator.device,
        )
        gathered_ids = self.accelerator.gather(unique_ids).cpu().numpy()
        hi = float(self.training_args.nft_query_group_loss_mask_high_reward_threshold)
        lo = float(self.training_args.nft_query_group_loss_mask_low_std_threshold)
        keep_mask_global, stats = compute_nft_query_group_loss_keep_mask_global(
            aggregated_rewards,
            gathered_ids,
            high_reward_threshold=hi,
            low_std_threshold=lo,
        )
        filtered_samples = int(stats["filtered_samples"])
        total_samples = int(stats["total_samples"])
        filtered_group_count = int(stats["filtered_groups"])
        total_groups = int(stats["total_groups"])

        if stats.get("reset_all"):
            logger.warning(
                "Query-group loss mask filtered all samples; disabling mask for this step to keep training stable."
            )

        if self.accelerator.is_main_process:
            logger.info(
                "Query-group loss mask: filtered %d/%d samples from %d/%d query groups "
                "(condition: mean>%s and std<%s).",
                filtered_samples,
                total_samples,
                filtered_group_count,
                total_groups,
                hi,
                lo,
            )

        if self.accelerator.is_main_process:
            self.log_data(
                {
                    "train/query_group_filtered_samples": float(filtered_samples),
                    "train/query_group_total_samples": float(total_samples),
                    "train/query_group_filtered_groups": float(filtered_group_count),
                    "train/query_group_total_groups": float(total_groups),
                },
                step=self.step,
            )

        gathered_len = int(keep_mask_global.shape[0])
        n_proc = int(self.accelerator.num_processes)
        if gathered_len % n_proc != 0:
            raise ValueError(
                f"Cannot reshape gathered loss mask of length {gathered_len} across "
                f"{n_proc} processes (remainder non-zero); check sampler / batch sizes."
            )
        per_rank = gathered_len // n_proc
        keep_mask = torch.as_tensor(keep_mask_global, dtype=torch.float32).reshape(
            n_proc,
            per_rank,
        )[int(self.accelerator.process_index)].to(self.accelerator.device)
        return keep_mask

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards, compute advantages, and log advantage metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): NFT matching loss with optional KL."""
        if self.training_args.nft_query_group_loss_mask:
            loss_keep_mask = self._build_query_group_loss_keep_mask(samples)
            for sample, keep in zip(samples, loss_keep_mask):
                sample.extra_kwargs["loss_keep_mask"] = float(keep.item())
        else:
            for sample in samples:
                sample.extra_kwargs.pop("loss_keep_mask", None)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]
            
            # Re-group samples into batches
            sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(shuffled_samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
            ]

            # ==================== Pre-compute: Timesteps, Noise, and Old V Predictions ====================
            self.adapter.rollout()
            with torch.no_grad(), self.autocast(), self.sampling_context():
                for batch in tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Pre-computing Old V Predictions',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]
                    
                    # Sample timesteps: (T, B)
                    all_timesteps = self._sample_timesteps(batch_size)
                    batch['_all_timesteps'] = all_timesteps
                    batch['_all_random_noise'] = [] # List[torch.Tensor]
                    
                    # Compute old v predictions with `sampling` policy
                    old_v_pred_list = []
                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]  # (B,) scheduler scale [0, 1000]
                        sigma_broadcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                        noise = randn_tensor(
                            clean_latents.shape,
                            device=clean_latents.device,
                            dtype=clean_latents.dtype,
                        )
                        batch['_all_random_noise'].append(noise)
                        noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise
                        old_output = self._compute_nft_output(batch, t_flat, noised_latents)
                        old_v_pred_list.append(old_output['noise_pred'].detach())
                    
                    batch['_old_v_pred_list'] = old_v_pred_list

            # ==================== Training Loop ====================
            self.adapter.train()
            loss_info = defaultdict(list)

            with self.autocast():
                for batch in tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Training',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    # Retrieve pre-computed data
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]
                    all_timesteps = batch['_all_timesteps']
                    all_random_noise = batch['_all_random_noise']
                    old_v_pred_list = batch['_old_v_pred_list']
                    # Iterate through timesteps
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Prepare inputs
                            t_flat = all_timesteps[t_idx]  # (B,) [0, 1000]
                            sigma_broadcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                            noise = all_random_noise[t_idx]
                            noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise
                            old_v_pred = old_v_pred_list[t_idx]
                            
                            # 2. Forward pass for current policy
                            output = self._compute_nft_output(batch, t_flat, noised_latents)
                            new_v_pred = output['noise_pred']
                            
                            # 3. Compute NFT loss
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            
                            # Normalize advantage to [0, 1]
                            normalized_adv = (adv / max(adv_clip_range)) / 2.0 + 0.5
                            r = torch.clamp(normalized_adv, 0, 1).view(-1, *([1] * (new_v_pred.dim() - 1)))
                            
                            # Positive/negative predictions
                            positive_pred = self.nft_beta * new_v_pred + (1 - self.nft_beta) * old_v_pred
                            negative_pred = (1.0 + self.nft_beta) * old_v_pred - self.nft_beta * new_v_pred
                            
                            # Positive loss
                            x0_pred = noised_latents - sigma_broadcast * positive_pred
                            with torch.no_grad():
                                weight = torch.abs(x0_pred.double() - clean_latents.double()).mean(
                                    dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                                ).clip(min=1e-5)
                            positive_loss = ((x0_pred - clean_latents) ** 2 / weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                            
                            # Negative loss
                            neg_x0_pred = noised_latents - sigma_broadcast * negative_pred
                            with torch.no_grad():
                                neg_weight = torch.abs(neg_x0_pred.double() - clean_latents.double()).mean(
                                    dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                                ).clip(min=1e-5)
                            negative_loss = ((neg_x0_pred - clean_latents) ** 2 / neg_weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                            
                            # Combined loss
                            ori_policy_loss = (r.squeeze() * positive_loss + (1.0 - r.squeeze()) * negative_loss) / self.nft_beta
                            weighted_policy_loss = ori_policy_loss * adv_clip_range[1]
                            keep_raw = batch.get("loss_keep_mask")
                            if keep_raw is None:
                                keep_mask = torch.ones(
                                    batch_size, device=adv.device, dtype=adv.dtype
                                )
                            else:
                                keep_mask = torch.as_tensor(
                                    keep_raw, device=adv.device, dtype=adv.dtype
                                ).reshape(-1)
                            if keep_mask.shape[0] != batch_size:
                                raise ValueError(
                                    f"loss_keep_mask length {keep_mask.shape[0]} != batch_size {batch_size}"
                                )
                            valid_count = torch.clamp(keep_mask.sum(), min=1.0)
                            policy_loss = (weighted_policy_loss * keep_mask).sum() / valid_count
                            loss = policy_loss
                            
                            # 4. KL penalty
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_output = self._compute_nft_output(batch, t_flat, noised_latents)
                                # KL-loss in v-space
                                kl_div = torch.mean(
                                    (new_v_pred - ref_output['noise_pred']) ** 2,
                                    dim=tuple(range(1, new_v_pred.ndim))
                                )
                                kl_loss = self.training_args.kl_beta * ((kl_div * keep_mask).sum() / valid_count)
                                loss = loss + kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            # 5. Log per-timestep info
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['unweighted_policy_loss'].append(ori_policy_loss.mean().detach())
                            loss_info['loss'].append(loss.detach())
                                
                            # 6. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Log loss info
                                loss_info = reduce_loss_info(self.accelerator, loss_info)
                                loss_info['grad_norm'] = grad_norm
                                self.log_data({f'train/{k}': v for k, v in loss_info.items()}, step=self.step)
                                self.step += 1
                                loss_info = defaultdict(list)