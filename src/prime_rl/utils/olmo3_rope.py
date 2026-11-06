"""Plain RoPE on Olmo 3's sliding-window layers, under the pinned transformers.

Olmo 3 uses YaRN on its full-attention layers only. transformers 5.6.2 reads the checkpoint's single
`rope_scaling` as one rotary for the whole model, so its 24 sliding layers (of 32) get YaRN too, and the
model writes degenerate text. vLLM has no native Olmo 3 at the pin and serves it through its Transformers
backend, so training and rollouts share the error, and mismatch KL does not show it. transformers 5.15
keeps a rotary per layer type. The inverse frequencies are computed here, not stored in a buffer: an extra
buffer would change the module's state for prime-rl's meta-device load and vLLM's weight loading.
"""

import torch
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer

_yarn_forward = Olmo3DecoderLayer.forward


def plain_rope(config, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Unscaled cos and sin for `position_ids`, which restart at each packed sample."""
    dim = config.hidden_size // config.num_attention_heads
    exponents = torch.arange(0, dim, 2, dtype=torch.int64, device=position_ids.device).float() / dim
    inv_freq = 1.0 / config.rope_parameters["rope_theta"] ** exponents
    freqs = position_ids[:, :, None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
             position_embeddings=None, **kwargs):
    if self.self_attn.attention_type == "sliding_attention":
        position_embeddings = plain_rope(self.self_attn.config, position_ids, hidden_states.dtype)
    return _yarn_forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, position_embeddings=position_embeddings, **kwargs)


def patch_olmo3_rope() -> None:
    Olmo3DecoderLayer.forward = _forward
