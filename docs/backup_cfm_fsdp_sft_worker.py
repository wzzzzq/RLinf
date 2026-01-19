# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import jax
import numpy as np
import torch
from omegaconf import DictConfig

import rlinf.algorithms  # noqa: F401
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
)


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.data_config = self.build_dataloader()
        self.data_iter = iter(self.data_loader)

    def init_worker(self):
        self.setup_model_and_optimizer()

        # Initialize EMA model for CFM training
        if hasattr(self.model, 'initialize_ema'):
            self.model.initialize_ema()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def build_dataloader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )

            # Check if episodes filtering is enabled
            episodes = self.cfg.data.get("episodes", None)
            if episodes is not None:
                episodes = list(episodes)
                return self._build_filtered_dataloader(config, episodes)
            else:
                data_loader = openpi_data_loader.create_data_loader(
                    config, shuffle=True
                )
                return data_loader, data_loader.data_config()
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _build_filtered_dataloader(self, config, episodes):
        """Build dataloader with episodes filtering using SubsetRandomSampler."""
        import logging

        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        import openpi.training.data_loader as openpi_data_loader
        import openpi.transforms as _transforms

        data_config = config.data.create(config.assets_dirs, config.model)
        logging.info(f"Building filtered dataloader with {len(episodes)} episodes")

        # Create full LeRobotDataset (without episodes filter to avoid network access)
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )

        # Build indices for target episodes
        episode_data_index = dataset.episode_data_index
        target_indices = []
        for ep_idx in episodes:
            if ep_idx < len(episode_data_index["from"]):
                start_idx = episode_data_index["from"][ep_idx].item()
                end_idx = episode_data_index["to"][ep_idx].item()
                target_indices.extend(range(start_idx, end_idx))
        logging.info(f"Dataset size after filtering: {len(target_indices)} frames")

        # Apply transforms (same as openpi)
        if data_config.prompt_from_task:
            dataset = openpi_data_loader.TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        dataset = openpi_data_loader.transform_dataset(
            dataset, data_config, skip_norm_stats=False
        )

        # Standard DataLoader with SubsetRandomSampler
        local_batch_size = config.batch_size // self._world_size if torch.distributed.is_initialized() else config.batch_size
        sampler = torch.utils.data.SubsetRandomSampler(target_indices)

        torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=False,  # sampler handles shuffling
            sampler=sampler,
            num_workers=0,
            drop_last=True,
            collate_fn=lambda items: jax.tree.map(
                lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
            ),
        )

        return _FilteredDataLoader(data_config, torch_loader), data_config

    def run_training(self):
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}

            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )
                observation, actions = next(self.data_iter)

                observation = jax.tree.map(
                    lambda x: torch.as_tensor(x, device=self.device)
                    .contiguous()
                    .clone(),
                    observation,
                )
                actions = actions.to(torch.float32)
                actions = actions.to(self.device)

                with self.amp_context:
                    losses = self.model(
                        data={"observation": observation, "actions": actions},
                        mode="sft",
                    )
                    if isinstance(losses, (list, tuple)):
                        losses = torch.stack(losses)
                    elif not isinstance(losses, torch.Tensor):
                        losses = torch.tensor(
                            losses, device=self.device, dtype=torch.float32
                        )
                    loss = losses.mean()

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Update EMA model for CFM training
            if hasattr(self.model, 'update_ema'):
                self.model.update_ema()

            # Collect stats
            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": loss.item(),
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    def set_global_step(self, global_step):
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)


class _FilteredDataLoader:
    """Wrapper for filtered dataloader with infinite iteration."""

    def __init__(self, data_config, torch_loader):
        self._data_config = data_config
        self._torch_loader = torch_loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        import openpi.models.model as _model

        while True:
            for batch in self._torch_loader:
                batch = jax.tree.map(torch.as_tensor, batch)
                yield _model.Observation.from_dict(batch), batch["actions"]
