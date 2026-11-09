"""Gemma 4's expert router without its in-place division.

`Gemma4TextRouter.forward` in transformers divides the output of `torch.topk` in place. Selective activation checkpointing
saves that output so routing repeats exactly on recompute, and it refuses a saved tensor that has been changed. The division
below is the same computation, out of place.
"""

import torch
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter


def _forward(self: Gemma4TextRouter, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states * self.scale * self.scalar_root_size

    expert_scores = self.proj(hidden_states)
    router_probabilities = nn.functional.softmax(expert_scores, dim=-1)

    top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

    return router_probabilities, top_k_weights, top_k_index


def patch_gemma4_router() -> None:
    Gemma4TextRouter.forward = _forward
