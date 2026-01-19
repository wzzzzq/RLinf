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

"""Latent Actor for DSRL (Diffusion Steering via RL).

This module implements a latent actor that predicts initial noise for
diffusion/flow matching policies. The latent actor is trained with PPO
to learn optimal initial noise distributions.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class LatentActor(nn.Module):
    """
    Latent Actor π^w(s) for DSRL.

    Predicts initial noise (mean and std) for diffusion/flow matching policies.
    Trained with PPO to maximize expected return.

    Args:
        state_dim: Dimension of state feature (prefix_output.mean + robot_state).
        action_horizon: Number of action steps in the chunk.
        action_dim: Dimension of each action.
        hidden_dims: List of hidden layer dimensions.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
        init_std: Initial standard deviation for noise prediction.
    """

    def __init__(
        self,
        state_dim: int = 2056,  # 2048 (prefix) + 8 (state for LIBERO)
        action_horizon: int = 50,
        action_dim: int = 24,
        hidden_dims: list[int] | None = None,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        init_std: float = 1.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.latent_dim = action_horizon * action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build trunk network
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.logstd_head = nn.Linear(hidden_dims[-1], self.latent_dim)

        # Initialize weights
        self._init_weights(init_std)

        # Register a buffer to store dtype for FSDP compatibility
        # FSDP replaces original parameters, so we can't use next(parameters()).dtype
        self.register_buffer("_dtype_buffer", torch.zeros(1), persistent=False)

    def _init_weights(self, init_std: float):
        """Initialize network weights for stable training."""
        # Initialize trunk with orthogonal initialization
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize mean head to output near-zero mean
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        # Initialize logstd head to output desired initial std
        nn.init.zeros_(self.logstd_head.weight)
        init_logstd = torch.log(torch.tensor(init_std))
        nn.init.constant_(self.logstd_head.bias, init_logstd.item())

    def forward(self, state_feat: torch.Tensor) -> Normal:
        """
        Forward pass to get noise distribution.

        Args:
            state_feat: State feature tensor of shape [bs, state_dim].

        Returns:
            Normal distribution over noise of shape [bs, action_horizon, action_dim].
        """
        # Use registered buffer dtype for FSDP compatibility
        # (FSDP replaces original parameters, making next(parameters()) fail)
        target_dtype = self._dtype_buffer.dtype
        if state_feat.dtype != target_dtype:
            state_feat = state_feat.to(target_dtype)

        # Trunk forward
        h = self.trunk(state_feat)

        # Get mean and std
        mean = self.mean_head(h)  # [bs, latent_dim]
        logstd = self.logstd_head(h)  # [bs, latent_dim]
        logstd = torch.clamp(logstd, self.log_std_min, self.log_std_max)

        # Reshape to [bs, action_horizon, action_dim]
        mean = mean.view(-1, self.action_horizon, self.action_dim)
        std = logstd.exp().view(-1, self.action_horizon, self.action_dim)

        return Normal(mean, std)

    def sample_and_log_prob(
        self,
        state_feat: torch.Tensor,
        action_chunk: int | None = None,
        action_env_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample noise and compute log probability.

        Args:
            state_feat: State feature tensor of shape [bs, state_dim].
            action_chunk: Number of action steps to compute logprob for (optional).
            action_env_dim: Action dimension for environment (optional).

        Returns:
            noise: Sampled noise of shape [bs, action_horizon, action_dim].
            log_prob: Log probability of sampled noise.
                If action_chunk and action_env_dim are provided, returns
                [bs, action_chunk, action_env_dim], otherwise [bs, action_horizon, action_dim].
        """
        dist = self.forward(state_feat)
        noise = dist.rsample()  # [bs, action_horizon, action_dim]
        log_prob = dist.log_prob(noise)  # [bs, action_horizon, action_dim]

        # Optionally clip to action_chunk and action_env_dim for PPO
        if action_chunk is not None and action_env_dim is not None:
            log_prob = log_prob[:, :action_chunk, :action_env_dim]

        return noise, log_prob

    def get_log_prob(
        self,
        state_feat: torch.Tensor,
        noise: torch.Tensor,
        action_chunk: int | None = None,
        action_env_dim: int | None = None,
    ) -> torch.Tensor:
        """
        Compute log probability of given noise samples.

        Args:
            state_feat: State feature tensor of shape [bs, state_dim].
            noise: Noise samples of shape [bs, action_horizon, action_dim].
            action_chunk: Number of action steps to compute logprob for (optional).
            action_env_dim: Action dimension for environment (optional).

        Returns:
            Log probability tensor.
                If action_chunk and action_env_dim are provided, returns
                [bs, action_chunk, action_env_dim], otherwise [bs, action_horizon, action_dim].
        """
        dist = self.forward(state_feat)
        log_prob = dist.log_prob(noise)  # [bs, action_horizon, action_dim]

        # Optionally clip to action_chunk and action_env_dim for PPO
        if action_chunk is not None and action_env_dim is not None:
            log_prob = log_prob[:, :action_chunk, :action_env_dim]

        return log_prob

    def get_entropy(
        self,
        state_feat: torch.Tensor,
        action_chunk: int | None = None,
        action_env_dim: int | None = None,
    ) -> torch.Tensor:
        """
        Compute entropy of the noise distribution.

        Args:
            state_feat: State feature tensor of shape [bs, state_dim].
            action_chunk: Number of action steps to compute entropy for (optional).
            action_env_dim: Action dimension for environment (optional).

        Returns:
            Entropy tensor.
        """
        dist = self.forward(state_feat)
        entropy = dist.entropy()  # [bs, action_horizon, action_dim]

        # Optionally clip to action_chunk and action_env_dim
        if action_chunk is not None and action_env_dim is not None:
            entropy = entropy[:, :action_chunk, :action_env_dim]

        return entropy
