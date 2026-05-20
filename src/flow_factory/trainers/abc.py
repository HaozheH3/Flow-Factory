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

# src/flow_factory/trainers/abc.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from ..hparams import *
from ..models.abc import BaseAdapter
from ..data_utils.loader import get_dataloader
from ..data_utils.dataset import (
    attach_per_sample_metadata_for_inference,
    materialize_jsonl_image_column_for_inference,
    _resolve_path,
)
from ..rewards import load_reward_model, BaseRewardModel, MultiRewardLoader, RewardProcessor, RewardBuffer
from ..advantage import AdvantageProcessor
from ..logger import load_logger, LogFormatter
from ..logger.formatting import (
    ROLLOUT_DUMP_EVAL_FLAT_INDEX_KEY,
    ROLLOUT_DUMP_GENERATED_IMAGE_ABS_KEY,
    ROLLOUT_DUMP_JUDGE_TRANSCRIPT_JSON_ABS_KEY,
    ROLLOUT_DUMP_SAMPLES_JSONL_ABS_KEY,
)
from ..samples import BaseSample, DPO_PREFERENCE_CANDIDATE_FLAG
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """
    def __init__(
            self,
            accelerator: Accelerator,
            config : Arguments,
            adapter : BaseAdapter,
        ):
        self.accelerator = accelerator
        self.config = config
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args

        self.reward_args = config.reward_args
        self.eval_reward_args = config.eval_reward_args or config.reward_args # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self._initialization()
        self.adapter.post_init()
        self._init_logging_backend()

        self._patch_deepspeed_autocast(accelerator)
        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    def _materialize_jsonl_images_for_adapter_inference(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        da = self.config.data_args
        out = materialize_jsonl_image_column_for_inference(
            kwargs,
            dataset_dir=da.dataset_dir,
            image_dir=da.image_dir,
        )
        return attach_per_sample_metadata_for_inference(
            out,
            inference_callable=self.adapter.inference,
        )

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    def _eval_samples_for_logger(self, samples: List[BaseSample]) -> List[BaseSample]:
        """Slice eval samples for WandB/SwanLab media; ``None`` cap keeps full list."""
        cap = self.log_args.log_max_eval_samples
        if cap is None:
            return samples
        if cap <= 0:
            return []
        return samples[:cap]

    def _predicted_rollout_artifact_output_dir_abs(self, subdir: str, leaf_dir_name: str) -> str:
        """Absolute path matching :func:`eval_dump.dump_eval_artifacts` ``output_dir`` layout."""
        run_name = "" if self.log_args.run_name is None else str(self.log_args.run_name)
        root = os.path.join(str(self.log_args.save_dir), run_name, subdir.strip())
        return os.path.abspath(os.path.join(root, leaf_dir_name))

    def _gather_object_flat_start_index(self, num_local_items: int) -> int:
        """Index of this rank's first sample in ``gather_object(local_sample_lists)`` order (ranks 0..R-1)."""
        if num_local_items < 0:
            raise ValueError(
                f"gather-object flat-start: num_local_items must be >= 0, got {num_local_items!r}"
            )
        nt = torch.tensor([int(num_local_items)], dtype=torch.long, device=self.accelerator.device)
        collected = self.accelerator.gather(nt)
        sizes = collected.detach().cpu().reshape(-1).tolist()
        proc = int(self.accelerator.process_index)
        nproc = int(self.accelerator.num_processes)
        if len(sizes) != nproc:
            raise RuntimeError(
                f"gather-object flat-start: expected accelerator.gather length {nproc}, got {len(sizes)} "
                f"(tensor shape {tuple(collected.shape)})"
            )
        if proc < 0 or proc >= len(sizes):
            raise RuntimeError(
                f"gather-object flat-start: invalid process_index={proc} for sizes len={len(sizes)}"
            )
        return int(sum(int(s) for s in sizes[:proc]))

    def _stamp_predicted_rollout_dump_eval_log_paths(
        self,
        logged_samples: List[BaseSample],
        *,
        flat_start_index: int,
        out_dir_abs: str,
    ) -> None:
        """Predicted paths for ``dump_eval_artifacts`` outputs (NDJSON row + optional judge JSON + PNG)."""
        out_root = os.path.abspath(out_dir_abs)
        img_root_abs = os.path.join(out_root, "images")
        jsonl_abs = os.path.join(out_root, "samples.jsonl")
        judge_root_abs = os.path.join(out_root, "judge_transcripts")
        for offset, sample in enumerate(logged_samples):
            if not isinstance(sample, BaseSample):
                raise TypeError(
                    f"rollout dump path stamp: expected BaseSample at offset {offset}, "
                    f"got {type(sample).__name__}: {sample!r}"
                )
            g = int(flat_start_index) + int(offset)
            gen_png = os.path.join(img_root_abs, f"sample_{g:05d}_generated.png")
            judge_json = os.path.join(judge_root_abs, f"sample_{g:05d}.json")
            sample.extra_kwargs[ROLLOUT_DUMP_GENERATED_IMAGE_ABS_KEY] = os.path.abspath(gen_png)
            sample.extra_kwargs[ROLLOUT_DUMP_SAMPLES_JSONL_ABS_KEY] = os.path.abspath(jsonl_abs)
            sample.extra_kwargs[ROLLOUT_DUMP_EVAL_FLAT_INDEX_KEY] = g
            sample.extra_kwargs[ROLLOUT_DUMP_JUDGE_TRANSCRIPT_JSON_ABS_KEY] = os.path.abspath(judge_json)

    def _stamp_eval_logged_samples_predicted_dump_png_paths_if_enabled(
        self,
        logged_eval_samples: List[BaseSample],
        *,
        flat_start_index: int,
        eval_status: str,
    ) -> None:
        """Main process only; matches :meth:`_gather_eval_samples_and_maybe_dump` leaf naming."""
        if not self.accelerator.is_main_process:
            return
        if not self.eval_args.eval_dump_enable:
            return
        if not str(self.log_args.save_dir).strip():
            return
        subdir = (self.eval_args.eval_dump_subdir or "eval_results").strip()
        leaf = f"epoch_{self.epoch:04d}_step_{self.step:06d}_{eval_status}"
        out_abs = self._predicted_rollout_artifact_output_dir_abs(subdir, leaf)
        self._stamp_predicted_rollout_dump_eval_log_paths(
            logged_eval_samples,
            flat_start_index=flat_start_index,
            out_dir_abs=out_abs,
        )

    def _maybe_stamp_train_samples_predicted_dump_png_paths(
        self,
        adv_metrics: Dict[str, Any],
        *,
        num_local_rollout_samples: int,
    ) -> None:
        """Append predicted dump paths to ``train_samples`` media when train dump runs this epoch."""
        if not self.training_args.train_dump_enable:
            return
        freq = self.training_args.train_dump_freq
        if freq < 1 or self.epoch % freq != 0:
            return
        if not str(self.log_args.save_dir).strip():
            return
        ts = adv_metrics.get("train_samples")
        if not isinstance(ts, list) or not ts:
            return
        flat_start = self._gather_object_flat_start_index(int(num_local_rollout_samples))
        subdir = (self.training_args.train_dump_subdir or "eval_results").strip()
        leaf = f"train_epoch_{self.epoch:04d}_step_{self.step:06d}"
        out_abs = self._predicted_rollout_artifact_output_dir_abs(subdir, leaf)
        self._stamp_predicted_rollout_dump_eval_log_paths(
            ts,
            flat_start_index=flat_start,
            out_dir_abs=out_abs,
        )

    def should_continue_training(self) -> bool:
        """Outer epoch loop: continue unless a finite ``max_epochs`` has been reached."""
        m = self.training_args.max_epochs
        if m is None or m < 0:
            return True
        return self.epoch < m

    def _format_console_scalar(self, k: str, v: float) -> str:
        """Format one scalar metric for the console summary line."""
        as_int = isinstance(v, int) or (isinstance(v, float) and v.is_integer())
        if as_int:
            return f"{k}={int(v)}"
        return f"{k}={v:.4f}"

    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)
        
        # Print summary to console
        if self.accelerator.is_local_main_process:
            metrics = {k: v for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items()) if v is not None}
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(self._format_console_scalar(k, v) for k, v in metrics.items())
                logger.info(" ".join(parts))
    
    def _init_logging_backend(self):
        """Initialize logging backend if specified."""
        if self.accelerator.is_main_process:
            self.logger = load_logger(self.config)
        else:
            self.logger = None
        self.accelerator.wait_for_everyone()

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # NOTE: This bug persists even with this context manager. DONOT USE ZeRO-3.
        # A possible solution: use DeepSpeed GatherParamter manually in the reward_model's `forward`.

        # Initialize all reward model instances
        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            eval_reward_args=self.config.eval_reward_args,
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        train_reward_configs = self.reward_loader.get_reward_configs('train')
        eval_reward_configs = self.reward_loader.get_reward_configs('eval')
        # Initialize reward processor
        group_on_same_rank = self.config.data_args.sampler_type == "group_contiguous"
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            reward_configs=train_reward_configs,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        self.eval_reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.eval_reward_models,
            reward_configs=eval_reward_configs,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        # Initialize reward buffers
        self.reward_buffer = RewardBuffer(
            self.reward_processor, self.training_args.group_size,
        )
        self.eval_reward_buffer = RewardBuffer(
            self.eval_reward_processor, self.training_args.group_size,
        )

        # Initialize advantage processor
        self.advantage_processor = AdvantageProcessor(
            accelerator=self.accelerator,
            reward_weights={
                name: cfg.weight
                for name, cfg in train_reward_configs.items()
            },
            group_size=self.training_args.group_size,
            global_std=getattr(self.training_args, 'global_std', True),
            sampler_type=self.config.data_args.sampler_type,
            verbose=self.log_args.verbose,
            max_train_samples_for_log=self.log_args.log_max_train_samples,
        )

        return self.reward_models, self.eval_reward_models

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Move text-encoder & vae to GPU for dataloader encoding
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules,
            device=self.accelerator.device
        )
        dataloader, test_dataloader = get_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        # Offload text-encoder after dataloader encoding
        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

        return dataloader, test_dataloader
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _load_inference_components(self, trainable_module_names: List[str]):
        """
        Load non-trainable components needed at runtime to the accelerator device.
        
        Trainable modules are already on-device via `accelerator.prepare()`.
        This loads the remaining modules required for inference and,
        when preprocessing is disabled, also loads encoding components
        that would otherwise stay offloaded.
        """
        prepared_names = set(trainable_module_names)
        
        modules_to_load = list(self.adapter.inference_modules)
        
        if not self.config.data_args.enable_preprocess:
            modules_to_load.extend(self.adapter.preprocessing_modules)
        
        # Resolve group names → concrete names, then deduplicate & exclude prepared
        resolved = self.adapter._resolve_component_names(modules_to_load)
        resolved = [m for m in resolved if m not in prepared_names]
        
        if resolved:
            self.adapter.on_load_components(
                components=resolved,
                device=self.accelerator.device,
            )

    def _initialization(self):
        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            # self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()

        # Init dataloader and optimizer
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Dynamically get all trainable modules from target_module_map
        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name) 
            for name in trainable_module_names 
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]
        # Prepare trainable modules + optimizer + test_dataloader
        to_prepare = trainable_modules + [self.optimizer]
        if self.test_dataloader is not None:
            to_prepare.append(self.test_dataloader)

        prepared = self.accelerator.prepare(*to_prepare)
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        self.optimizer = prepared[len(trainable_modules)]
        if self.test_dataloader is not None:
            self.test_dataloader = prepared[len(trainable_modules) + 1]

        # Load inference modules, excluding already-prepared ones
        self._load_inference_components(trainable_module_names)
        
        # Initialize reward model
        self._init_reward_model()

    def _synchronize_frozen_components(self):
        if self.accelerator.num_processes <= 1:
            return
        
        # Synchronize all non-prepared components
        all_names = self.adapter._resolve_component_names()
        for name in all_names:
            if self.adapter._should_manage_device(name):
                comp = self.adapter.get_component(name)
                if comp is not None:
                    for param in comp.parameters():
                        param.data = param.data.to(self.accelerator.device)
                        dist.broadcast(param.data, src=0)

        # Barrier to ensure everyone is done
        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {self.accelerator.process_index}] Frozen components synchronized.")

    @staticmethod
    def _patch_deepspeed_autocast(accelerator):
        """Patch DeepSpeed >=0.17.2 to allow external torch.autocast contexts.

        In v0.17.2+, engine.forward() calls validate_nested_autocast() which
        raises AssertionError if torch.autocast is active outside the engine,
        then wraps the forward with torch.autocast(enabled=torch_autocast_enabled).
        When torch_autocast is not configured (the default for bf16 built-in
        mixed-precision), this inner context uses enabled=False, which explicitly
        *disables* any outer autocast and causes dtype mismatches.

        This patch makes the engine transparent to an outer autocast context:
        validate_nested_autocast becomes a no-op, and torch_autocast_enabled /
        torch_autocast_dtype fall through to the active torch.autocast state so
        the engine re-enables (rather than disables) autocast during forward.
        """
        if getattr(accelerator.state, 'deepspeed_plugin', None) is None:
            return

        try:
            import deepspeed.runtime.torch_autocast as _ds_ac
            from deepspeed.runtime.engine import DeepSpeedEngine
        except ImportError:
            return

        if getattr(DeepSpeedEngine, '_ff_autocast_patched', False):
            return

        if hasattr(_ds_ac, 'validate_nested_autocast'):
            _ds_ac.validate_nested_autocast = lambda engine: None

        if hasattr(DeepSpeedEngine, 'torch_autocast_enabled'):
            _orig_enabled = DeepSpeedEngine.torch_autocast_enabled
            _orig_dtype = DeepSpeedEngine.torch_autocast_dtype

            def _patched_enabled(self):
                return _orig_enabled(self) or torch.is_autocast_enabled()

            def _patched_dtype(self):
                if not _orig_enabled(self) and torch.is_autocast_enabled():
                    return torch.get_autocast_gpu_dtype()
                return _orig_dtype(self)

            DeepSpeedEngine.torch_autocast_enabled = _patched_enabled
            DeepSpeedEngine.torch_autocast_dtype = _patched_dtype

        DeepSpeedEngine._ff_autocast_patched = True

    @abstractmethod
    def start(self, *args, **kwargs):
        """Start training process."""
        pass

    @abstractmethod
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Stages 4--5: finalize rewards, compute advantages, and log metrics (no policy gradients).

        Algorithms that need extra batching before the loss (e.g. DPO chosen/rejected pairs) may
        perform that work in :meth:`optimize` after advantages are on each sample.
        """
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation for one epoch."""
        pass

    def _gather_reward_arrays_from_local_samples(
        self, local_samples: List[BaseSample]
    ) -> Dict[str, np.ndarray]:
        """Stack per-sample rewards (and advantage when present), ``accelerator.gather`` like eval."""

        def _scalar_reward_component(x: Any, *, component: str, sample_index: int) -> torch.Tensor:
            device = self.accelerator.device
            if isinstance(x, torch.Tensor):
                t = x.detach().to(device).reshape(-1)
            else:
                t = torch.as_tensor(x, device=device).reshape(-1)
            if t.numel() != 1:
                raise ValueError(
                    f"train rollout dump: expected scalar-like {component} for sample "
                    f"{sample_index}, got shape {tuple(t.shape)}"
                )
            return t[0]

        if not local_samples:
            self.accelerator.wait_for_everyone()
            return {}

        first_extras = local_samples[0].extra_kwargs
        reward_keys = sorted((first_extras.get("rewards") or {}).keys())
        include_advantage = "advantage" in first_extras

        tensors: Dict[str, torch.Tensor] = {}
        for k in reward_keys:
            vals: List[torch.Tensor] = []
            for i, s in enumerate(local_samples):
                rmap = s.extra_kwargs.get("rewards")
                if rmap is None or k not in rmap:
                    raise ValueError(
                        "train rollout dump: sample "
                        f"{i} missing extra_kwargs['rewards'][{k!r}] after prepare_feedback."
                    )
                vals.append(_scalar_reward_component(rmap[k], component=f"rewards[{k!r}]", sample_index=i))
            tensors[k] = torch.stack(vals)

        if include_advantage:
            adv_vals: List[torch.Tensor] = []
            for i, s in enumerate(local_samples):
                if "advantage" not in s.extra_kwargs:
                    raise ValueError(
                        f"train rollout dump: sample {i} missing extra_kwargs['advantage'] "
                        "after prepare_feedback."
                    )
                adv_vals.append(
                    _scalar_reward_component(
                        s.extra_kwargs["advantage"],
                        component="advantage",
                        sample_index=i,
                    )
                )
            tensors["advantage"] = torch.stack(adv_vals)

        gathered: Dict[str, np.ndarray] = {}
        for key, value in tensors.items():
            gathered_tensor: torch.Tensor = self.accelerator.gather(value)  # type: ignore[assignment]
            gathered[key] = gathered_tensor.cpu().numpy()
        self.accelerator.wait_for_everyone()
        return gathered

    def _gather_rollout_artifacts_and_maybe_dump(
        self,
        local_samples: List[BaseSample],
        gathered_rewards: Dict[str, np.ndarray],
        *,
        subdir: str,
        leaf_dir_name: str,
        log_kind: str,
        summary_status: str,
        error_traceback: Optional[str] = None,
    ) -> None:
        """Gather samples with ``gather_object``; main process writes eval_dump-compatible artifacts."""
        from accelerate.utils.operations import gather_object

        from ..utils.eval_dump import dump_eval_artifacts

        # accelerate ``gather_object`` on a list returns one flat list across ranks (concat per rank).
        # Do not iterate samples again: ``BaseSample.__iter__`` yields field names, not samples.
        gathered_samples = gather_object(local_samples)
        if self.accelerator.is_main_process:
            flat: List[BaseSample] = []
            for i, item in enumerate(gathered_samples):
                if not isinstance(item, BaseSample):
                    raise TypeError(
                        "rollout gather_object expected only BaseSample instances after concat "
                        f"across ranks; index {i} has type {type(item).__name__}: {item!r}"
                    )
                flat.append(item)
            run_name = str(self.log_args.run_name)
            root = os.path.join(str(self.log_args.save_dir), run_name, subdir)
            out_dir = os.path.join(root, leaf_dir_name)
            logger.info(
                "%s: writing summary/images under %s "
                "(separate from reward frontier_judge_dump_dir ToolGen/Frontier logs)",
                log_kind,
                out_dir,
            )
            dump_eval_artifacts(
                output_dir=out_dir,
                epoch=self.epoch,
                step=self.step,
                samples=flat,
                gathered_rewards=gathered_rewards,
                status=summary_status,
                error_traceback=error_traceback,
            )
        self.accelerator.wait_for_everyone()

    def _gather_eval_samples_and_maybe_dump(
        self,
        local_samples: List[BaseSample],
        gathered_rewards: Dict[str, np.ndarray],
        *,
        status: str,
        error_traceback: Optional[str] = None,
    ) -> None:
        """Gather eval samples across ranks; main process may write artifacts under the run folder."""
        if not self.eval_args.eval_dump_enable:
            return

        save_dir = self.log_args.save_dir
        if not save_dir:
            if self.accelerator.is_main_process:
                logger.warning(
                    "eval_dump_enable is True but log.save_dir is empty; skipping eval artifact dump."
                )
            return

        subdir = (self.eval_args.eval_dump_subdir or "eval_results").strip()
        self._gather_rollout_artifacts_and_maybe_dump(
            local_samples,
            gathered_rewards,
            subdir=subdir,
            leaf_dir_name=f"epoch_{self.epoch:04d}_step_{self.step:06d}_{status}",
            log_kind="Eval snapshot (eval_dump_enable)",
            summary_status=status,
            error_traceback=error_traceback,
        )

    def _gather_train_samples_and_maybe_dump(self, local_samples: List[BaseSample]) -> None:
        """After prepare_feedback: optionally dump training rollouts next to eval_results-style paths."""
        if not self.training_args.train_dump_enable:
            return
        freq = self.training_args.train_dump_freq
        if freq < 1 or self.epoch % freq != 0:
            return

        save_dir = self.log_args.save_dir
        if not save_dir:
            if self.accelerator.is_main_process:
                logger.warning(
                    "train_dump_enable is True but log.save_dir is empty; skipping train rollout dump."
                )
            self.accelerator.wait_for_everyone()
            return

        gathered_rewards = self._gather_reward_arrays_from_local_samples(local_samples)
        subdir = (self.training_args.train_dump_subdir or "eval_results").strip()
        self._gather_rollout_artifacts_and_maybe_dump(
            local_samples,
            gathered_rewards,
            subdir=subdir,
            leaf_dir_name=f"train_epoch_{self.epoch:04d}_step_{self.step:06d}",
            log_kind="Training rollout snapshot (train_dump_enable)",
            summary_status="train",
            error_traceback=None,
        )

    def _maybe_offload_samples_to_cpu(self, samples: List[BaseSample]) -> None:
        """Move every sample's tensor fields to CPU when ``offload_samples_to_cpu`` is enabled.

        Producer-side half of the CPU-offload + lazy-reload pipeline: samples
        leave ``sample()`` already on CPU so that the GPU peak from the rollout
        buffer is bounded by a single batch worth of inference activations.

        Must be called BEFORE ``self.reward_buffer.add_samples(...)`` so that
        the buffer's recorded ``sync_event`` captures "D2H complete + data
        ready on CPU"; downstream reward workers (sync or async) then see a
        deterministic CPU-resident state and trigger their own H2D inside
        ``RewardProcessor`` (see ``move_tensors_to_device`` in
        ``utils/base.py``).

        No-op when ``training_args.offload_samples_to_cpu`` is False
        (default), preserving the legacy GPU-resident behaviour.

        Args:
            samples: Newly generated samples for the current sample loop iteration.
        """
        if not self.training_args.offload_samples_to_cpu:
            return
        for sample in samples:
            sample.to('cpu')

    def _inject_preference_candidates(self, sample_batch: List[BaseSample]) -> List[BaseSample]:
        """Append preference-image candidates after a complete rollout group.

        ``sample_batch`` must contain ``group_size`` policy rollouts sharing the same
        ``sample.unique_id``. This method emits those K samples followed by
        ``preference_extra_candidates`` packed clones.

        Requires the adapter to implement ``dpo_clone_sample_with_preference_image``.
        """
        n_extra = getattr(self.training_args, 'preference_extra_candidates', 0)
        if n_extra <= 0:
            return sample_batch
        key = self.training_args.preference_candidate_metadata_key
        base_dir = self.config.data_args.dataset_dir
        expected_k = int(self.training_args.group_size)
        if len(sample_batch) != expected_k:
            raise ValueError(
                f"`_inject_preference_candidates` expected exactly "
                f"group_size ({expected_k}) rollout samples before preference injection; "
                f"got {len(sample_batch)}."
            )
        group_uid = sample_batch[0].unique_id
        if any(s.unique_id != group_uid for s in sample_batch[1:]):
            raise ValueError(
                "Preference injection received a rollout list whose members do not "
                f"share one `unique_id`; unique_ids={sorted({s.unique_id for s in sample_batch})!r}"
            )

        template = sample_batch[0]
        raw_paths = template.extra_kwargs.get(key)
        if raw_paths is None:
            raise KeyError(
                f"preference_extra_candidates={n_extra} requires extra_kwargs[{key!r}] on each "
                f"rollout sample (unique_id={group_uid})."
            )
        if not isinstance(raw_paths, list):
            raise TypeError(
                f"{key!r} must be a list of image path strings, got {type(raw_paths).__name__} "
                f"(unique_id={group_uid})"
            )
        if len(raw_paths) != n_extra:
            raise ValueError(
                f"{key!r} must have length {n_extra} (= preference_extra_candidates), "
                f"got {len(raw_paths)} for unique_id={group_uid}."
            )

        out: List[BaseSample] = list(sample_batch)
        for rel in raw_paths:
            if not isinstance(rel, str) or not rel.strip():
                raise ValueError(
                    f"Invalid image path in {key!r}: {rel!r} (unique_id={group_uid})"
                )
            path = _resolve_path(base_dir, rel)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Preference candidate image not found: {path!r} (from {rel!r}, unique_id={group_uid})"
                )
            pil = Image.open(path).convert("RGB")
            out.append(
                self.adapter.dpo_clone_sample_with_preference_image(
                    template, pil, key
                )
            )
        return out

    @staticmethod
    def _is_preference_candidate_sample(sample: BaseSample) -> bool:
        return sample.extra_kwargs.get(DPO_PREFERENCE_CANDIDATE_FLAG) is True

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save trainer state to a specific path."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        self.adapter.save_checkpoint(
            save_directory=save_directory,
            model_only=self.log_args.save_model_only,
        )

        self.accelerator.wait_for_everyone()

    def load_checkpoint(
            self,
            path: str,
            resume_type: Optional[Literal['lora', 'full', 'state']] = None,
        ):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            resume_type=resume_type,
        )
        self.accelerator.wait_for_everyone()