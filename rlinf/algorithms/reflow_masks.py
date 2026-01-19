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

"""Reflow mask strategies for RFC (Rectified Flow Consistency) loss."""

from typing import Optional

import torch

from rlinf.algorithms.registry import get_reflow_mask, register_reflow_mask


@register_reflow_mask("success_only")
def compute_success_only_mask(
    terminations: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Mask based on episode success (termination=True).
    Only successful episodes participate in reflow training.

    Args:
        terminations: Shape [bsz, num_action_chunks]
        advantages: Used only to determine output shape if terminations is None

    Returns:
        Boolean mask of shape [bsz, num_action_chunks]
    """
    if terminations is None:
        # If no terminations info, include all data
        if advantages is not None:
            return torch.ones_like(advantages, dtype=torch.bool)
        raise ValueError("Either terminations or advantages must be provided")

    # Check if any action chunk in the sample has termination (success)
    # terminations shape: [bsz, num_action_chunks]
    episode_success = terminations.any(dim=-1, keepdim=True)  # [bsz, 1]

    # Expand to full shape
    mask = episode_success.expand_as(terminations)  # [bsz, num_action_chunks]

    return mask


@register_reflow_mask("advantage_positive")
def compute_advantage_positive_mask(
    advantages: torch.Tensor,
    terminations: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Mask based on positive advantage values.
    Only steps/actions with advantage > 0 participate in reflow training.

    Args:
        advantages: Shape [bsz, num_action_chunks]
        terminations: Not used, kept for interface consistency

    Returns:
        Boolean mask of shape [bsz, num_action_chunks]
    """
    return advantages > 0


@register_reflow_mask("advantage_topk")
def compute_advantage_topk_mask(
    advantages: torch.Tensor,
    terminations: Optional[torch.Tensor] = None,
    topk_ratio: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """
    Mask based on top-K% advantage values.
    Only top-K% steps/actions by advantage ranking participate in reflow training.

    Args:
        advantages: Shape [bsz, num_action_chunks]
        terminations: Not used, kept for interface consistency
        topk_ratio: Float in (0, 1], fraction of data to keep

    Returns:
        Boolean mask of shape [bsz, num_action_chunks]
    """
    assert 0 < topk_ratio <= 1.0, f"topk_ratio must be in (0, 1], got {topk_ratio}"

    # Flatten advantages for global ranking
    flat_advantages = advantages.reshape(-1)
    total_elements = flat_advantages.numel()
    k = max(1, int(total_elements * topk_ratio))

    # Get threshold value at top-K position
    topk_values, _ = torch.topk(flat_advantages, k)
    threshold = topk_values[-1]  # Smallest value in top-K

    # Create mask for elements >= threshold
    mask = advantages >= threshold

    return mask


def compute_reflow_mask(
    mask_type: str,
    advantages: torch.Tensor,
    terminations: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Unified entry for reflow mask computation.

    Args:
        mask_type: Type of mask ("success_only", "advantage_positive", "advantage_topk")
        advantages: Shape [bsz, num_action_chunks]
        terminations: Shape [bsz, num_action_chunks], optional
        **kwargs: Additional parameters (e.g., topk_ratio for advantage_topk)

    Returns:
        Boolean tensor of same shape as advantages
    """
    fn = get_reflow_mask(mask_type)
    return fn(advantages=advantages, terminations=terminations, **kwargs)
