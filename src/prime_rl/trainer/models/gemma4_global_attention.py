"""Gemma 4's global attention layers under the Hugging Face path.

Global layers have head_dim 512 (`global_head_dim`). FlashAttention 2 stops at 256, and FlashAttention 4's backward at
(512, 512) requires block-sparsity inputs. So layers above FlashAttention 2's limit run PyTorch's scaled dot-product
attention, one packed sequence at a time; every other layer keeps FlashAttention 2.

The result is what `_flash_attention_forward` computes on a packed batch: sequences start where position_ids is 0 (the
padding at a bin's end is its own sequence), the score scale is the module's `scaling`, and the output is
[batch, seq, heads, dim].
"""

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

FLASH_ATTENTION_2_MAX_HEAD_DIM = 256


def packed_sdpa_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [batch, heads, seq, dim]
    key: torch.Tensor,  # [batch, kv_heads, seq, dim]
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    is_causal: bool | None = None,
    position_ids: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, "packed sdpa expects padding-free packed bins, as FlashAttention gets from prime-rl"
    assert sliding_window is None, "packed sdpa is for global layers"
    assert position_ids is not None, "packed sdpa needs position_ids for the sequence boundaries"
    batch, heads, seq, dim = query.shape
    groups = module.num_key_value_groups
    causal = module.is_causal if is_causal is None else is_causal

    q = query.transpose(1, 2).reshape(batch * seq, heads, dim)
    k = key.repeat_interleave(groups, dim=1).transpose(1, 2).reshape(batch * seq, heads, dim)
    v = k if value is key else value.repeat_interleave(groups, dim=1).transpose(1, 2).reshape(batch * seq, heads, dim)
    starts = (position_ids.reshape(-1) == 0).nonzero().view(-1).tolist()
    assert starts and starts[0] == 0, "position_ids must start each packed row at 0"

    outputs = []
    for start, end in zip(starts, starts[1:] + [batch * seq]):
        attended = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1).unsqueeze(0),
            k[start:end].transpose(0, 1).unsqueeze(0),
            v[start:end].transpose(0, 1).unsqueeze(0),
            dropout_p=dropout,
            is_causal=causal,
            scale=scaling,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    return torch.cat(outputs).view(batch, seq, heads, dim), None


def route_gemma4_global_attention() -> None:
    """Send attention layers above FlashAttention 2's head size to `packed_sdpa_forward`."""
    flash_attention = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    if getattr(flash_attention, "routes_global_layers", False):
        return

    def attention(module, query, key, value, attention_mask, **kwargs):
        if getattr(module, "head_dim", 0) > FLASH_ATTENTION_2_MAX_HEAD_DIM:
            return packed_sdpa_forward(module, query, key, value, attention_mask, **kwargs)
        return flash_attention(module, query, key, value, attention_mask, **kwargs)

    attention.routes_global_layers = True
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", attention)
