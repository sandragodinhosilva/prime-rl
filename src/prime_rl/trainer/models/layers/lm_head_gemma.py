from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.trainer.models.layers.lm_head import (
    IGNORE_INDEX,
    PrimeLmOutput,
    _online_logsumexp_and_weighted_update,
    _patch_model_forward,
    _sampling_mask_local_indices,
    cross_entropy_sum,
    sampling_replay_mask,
)
from prime_rl.utils.logger import get_logger


class GemmaFusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.softcap = softcap

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "GemmaFusedOutputLinear requires labels for chunked logprob computation"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()

        if temperature is None:
            assert sampling_mask is None, "sampling-mask replay requires per-token temperatures"
            # Summed cross-entropy through the logprob path; IGNORE_INDEX rows are masked at the sum.
            inv_t = torch.ones(b * s, device=hidden_states.device, dtype=torch.float32)
            logprobs, _ = _GemmaChunkedLogProbEntropyFn.apply(
                hidden_states, self.weight, labels, inv_t, self.chunk_size, self.softcap, None
            )
            return PrimeLmOutput(loss=-logprobs[labels != IGNORE_INDEX].sum())

        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]
        if sampling_mask is not None:
            sampling_mask = sampling_mask.reshape(b * s, sampling_mask.shape[-1]).contiguous()
        logprobs, entropy = _GemmaChunkedLogProbEntropyFn.apply(
            hidden_states, self.weight, labels, inv_t, self.chunk_size, self.softcap, sampling_mask
        )

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class GemmaVanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
    ) -> PrimeLmOutput:
        logits = super().forward(hidden_states)
        logits = self.softcap * torch.tanh(logits / self.softcap)
        if labels is not None and temperature is None:
            return PrimeLmOutput(loss=cross_entropy_sum(logits, labels))
        return PrimeLmOutput(logits=logits)


class _GemmaChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
        softcap: float,
        sampling_mask: torch.Tensor | None = None,  # [N, K] int32, -1-padded
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-token logprobs and entropy by chunking over flattened sequence tokens.

        Positions with a usable ``sampling_mask`` (see ``sampling_replay_mask``) are renormalized over that
        mask, on the softcapped and temperature-scaled logits. Other positions, and entropy, keep full-vocab
        normalization, as in ``_SequenceChunkedLogProbEntropyFn``.
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert inv_temperature.dim() == 1, f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert chunk_size > 0
        if sampling_mask is not None:
            assert sampling_mask.dim() == 2 and sampling_mask.shape[0] == hidden.shape[0], (
                f"expected sampling_mask [N,K], got {tuple(sampling_mask.shape)}"
            )

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)
        replay = torch.zeros((n,), device=device, dtype=torch.bool) if sampling_mask is not None else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            token_count = end - start

            m = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((token_count,), device=device, dtype=torch.float32)
            t = torch.zeros((token_count,), device=device, dtype=torch.float32)
            target_logits = torch.zeros((token_count,), device=device, dtype=torch.float32)

            mask_chunk = sampling_mask[start:end].to(torch.long) if sampling_mask is not None else None
            if mask_chunk is not None:
                replay_chunk = sampling_replay_mask(mask_chunk, labels_chunk)
                mask_logits = torch.full_like(mask_chunk, float("-inf"), dtype=torch.float32)

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32)
                scaled_logits = softcap * torch.tanh(scaled_logits / softcap)
                scaled_logits = scaled_logits * inv_t_chunk

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)

                if mask_chunk is not None:
                    mask_local, mask_in_range = _sampling_mask_local_indices(mask_chunk, vocab_start, vocab_end)
                    mask_logits = torch.where(mask_in_range, scaled_logits.gather(1, mask_local), mask_logits)

                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    target_logits[mask] = scaled_logits[mask, idx]

            logz_full = m + torch.log(s)
            if mask_chunk is not None:
                logz_chunk = torch.where(replay_chunk, torch.logsumexp(mask_logits, dim=-1), logz_full)
                replay[start:end] = replay_chunk
            else:
                logz_chunk = logz_full
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_full - (t / s)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz, sampling_mask, replay)
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in GemmaFusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature, logz, sampling_mask, replay = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        softcap: float = ctx.softcap

        n, _ = hidden.shape
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]
            mask_chunk = sampling_mask[start:end].to(torch.long) if sampling_mask is not None else None
            replay_chunk = replay[start:end] if replay is not None else None

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                logits_f = logits_chunk.to(torch.float32)
                tanh_val = torch.tanh(logits_f / softcap)
                scaled_logits = softcap * tanh_val
                scaled_logits = scaled_logits * inv_t_chunk
                if mask_chunk is not None:
                    # Replayed rows get softmax gradient only on mask ids; -inf before exp avoids overflow, as in
                    # _SequenceChunkedLogProbEntropyFn.
                    local, in_range = _sampling_mask_local_indices(mask_chunk, vocab_start, vocab_end)
                    mask_indicator = torch.zeros(scaled_logits.shape, dtype=torch.int8, device=scaled_logits.device)
                    mask_indicator.scatter_add_(1, local, in_range.to(torch.int8))
                    masked_out = replay_chunk.unsqueeze(-1) & (mask_indicator == 0)
                    scaled_logits.masked_fill_(masked_out, float("-inf"))
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    grad_logits[mask, idx] += grad_chunk[mask]
                grad_logits = grad_logits * inv_t_chunk
                grad_logits = grad_logits * (1 - tanh_val**2)

                grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None, None, None


def inject_gemma_lm_head(model: nn.Module, chunk_size: int | None, softcap: float) -> None:
    logger = get_logger()
    logger.info(f"Injecting Gemma LM head with chunk size {chunk_size}, softcap={softcap}")

    old_lm_head = model.lm_head
    if chunk_size is not None:
        model.lm_head = GemmaFusedOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            chunk_size=chunk_size,
            softcap=softcap,
        )
    else:
        model.lm_head = GemmaVanillaOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            softcap=softcap,
        )
    model.lm_head.weight = old_lm_head.weight
    del old_lm_head

    _patch_model_forward(model)
