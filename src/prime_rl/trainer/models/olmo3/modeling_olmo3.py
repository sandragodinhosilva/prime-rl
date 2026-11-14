"""PrimeRL implementation of Olmo 3 (dense; AI2's Olmo-3 / Olmo-3.1 Instruct and Think).

Mirrors transformers' ``modeling_olmo3`` (5.20) in its numerics, with PrimeRL's packed flash-attention and LM head:
- post-norm decoder layers: ``h = x + post_attention_layernorm(attn(x))``, ``out = h + post_feedforward_layernorm(mlp(h))``;
- QK-norm over the whole projection (``q_norm`` over heads x head_dim, ``k_norm`` over kv heads x head_dim), before RoPE;
- one rotary per layer type: the sliding-window layers use plain RoPE, the full-attention layers YaRN (``rope_parameters``
  keyed by layer type, as transformers >= 5.15 reads Olmo 3 configs); cos/sin stay in fp32 and the rotated q/k are cast back
  to the activations' dtype, as transformers does;
- RMSNorm multiplies by its weight in fp32 before casting back (Olmo's order; Llama's casts first).
Weights use transformers' names, so the HF <-> PrimeRL conversion is the identity.
"""

import functools
from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class Olmo3RMSNorm(nn.Module):
    """transformers' Olmo3RMSNorm: normalize in fp32, multiply by the weight in fp32, then cast back."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def _default_rope(config: Olmo3Config, device=None, layer_type: str | None = None, **kwargs) -> tuple[Tensor, float]:
    base = config.rope_parameters[layer_type]["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq.to(device), 1.0


def _per_layer_type_rope_parameters(config: Olmo3Config) -> dict:
    """``rope_parameters`` keyed by layer type. transformers >= 5.15 stores Olmo 3's that way; an older flat dict (one rotary
    for the whole model, the transformers-5.6 reading that gave the sliding layers YaRN) is refused rather than guessed."""
    params = config.rope_parameters
    if not (isinstance(params, dict) and all(lt in params for lt in set(config.layer_types))):
        raise ValueError(
            f"Olmo 3 needs rope_parameters keyed by layer type ({sorted(set(config.layer_types))}), got {params!r}: "
            "upgrade transformers (>= 5.15 reads Olmo 3 configs per layer type)"
        )
    return params


class Olmo3RotaryEmbedding(nn.Module):
    """One rotary per layer type (transformers' Olmo3RotaryEmbedding): fp32 cos/sin, scaled by the type's attention factor."""

    def __init__(self, config: Olmo3Config, device=None):
        super().__init__()
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        params = _per_layer_type_rope_parameters(config)
        self.rope_type = {lt: params[lt]["rope_type"] for lt in self.layer_types}
        for layer_type in self.layer_types:
            inv_freq, scaling = self._init_fn(layer_type)(config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", scaling)

    def _init_fn(self, layer_type: str):
        rope_type = self.rope_type[layer_type]
        return _default_rope if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]

    def reset_buffers(self) -> None:
        """Recompute the inverse frequencies (the model is built on the meta device)."""
        for layer_type in self.layer_types:
            buf = getattr(self, f"{layer_type}_inv_freq")
            inv_freq, scaling = self._init_fn(layer_type)(self.config, buf.device, layer_type=layer_type)
            buf.copy_(inv_freq)
            setattr(self, f"{layer_type}_attention_scaling", scaling)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor, layer_type: str) -> tuple[Tensor, Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        scaling = getattr(self, f"{layer_type}_attention_scaling")
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos() * scaling, emb.sin() * scaling


def apply_rotary_pos_emb_fp32(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """transformers' Olmo 3 rotary: q/k times fp32 cos/sin, then back to their dtype. q/k are [batch, heads, seq, dim]."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class Olmo3Attention(FlashAttention):
    """PrimeRL flash attention with Olmo 3's whole-projection QK-norm (transformers' RMSNorm order) and fp32 rotary."""

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__(config, flash_attn_version=flash_attn_version)
        self.q_norm = Olmo3RMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states = self.q_norm(query_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_norm(key_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_fp32(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.view(hidden_shape)
        out = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)
        attn_output = out.contiguous().view(1, out.shape[0], -1)
        return self.o_proj(attn_output), None


OLMO3_ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(Olmo3Attention, flash_attn_version=2),
    "flash_attention_3": functools.partial(Olmo3Attention, flash_attn_version=3),
    "flash_attention_4": functools.partial(Olmo3Attention, flash_attn_version=4),
}


class Olmo3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            output_bias=config.attention_bias,
            use_qk_norm=False,  # Olmo3Attention adds its own whole-projection norms
            rms_norm_eps=config.rms_norm_eps,
        )
        if config._attn_implementation not in OLMO3_ATTN_IMPL2CLASS:
            raise ValueError(f"Olmo 3 runs on flash attention only, got {config._attn_implementation!r}")
        self.self_attn = OLMO3_ATTN_IMPL2CLASS[config._attn_implementation](attn_config)
        if self.layer_type == "sliding_attention":
            self.self_attn.sliding_window = config.sliding_window
        self.mlp = FeedForward(dim=config.hidden_size, hidden_dim=config.intermediate_size, activation=config.hidden_act)
        self.post_attention_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.post_feedforward_layernorm(self.mlp(hidden_states))
        return residual + hidden_states


@auto_docstring
class Olmo3PreTrainedModel(PreTrainedModelPrimeRL):
    config: Olmo3Config
    base_model_prefix = "model"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Same key names in HF and PrimeRL format (see Qwen3PreTrainedModel).
        return False

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


@auto_docstring
class Olmo3Model(Olmo3PreTrainedModel):
    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Olmo3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Olmo3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        r"""
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract).
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        hidden_states = inputs_embeds
        position_embeddings = {
            lt: self.rotary_emb(hidden_states, position_ids, lt) for lt in self.rotary_emb.layer_types
        }
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[decoder_layer.layer_type],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class Olmo3ForCausalLM(Olmo3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Olmo3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract).
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Unused (HuggingFace API compatibility).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for PrimeRL's wrapped LM head (per-token logprobs/entropy).
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy when `labels` are given.
        """
        assert use_cache is None, "use_cache is not supported for custom olmo3"
        assert past_key_values is None, "past_key_values is not supported for custom olmo3"
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        self.model.rotary_emb.reset_buffers()


__all__ = ["Olmo3ForCausalLM", "Olmo3Model", "Olmo3PreTrainedModel"]
