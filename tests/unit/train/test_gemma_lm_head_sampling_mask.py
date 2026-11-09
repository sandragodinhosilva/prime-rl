"""GemmaFusedOutputLinear against a direct autograd computation: softcap, temperature, sampling-mask renormalization."""

import torch

from prime_rl.trainer.models.layers.lm_head_gemma import GemmaFusedOutputLinear

SOFTCAP = 30.0
VOCAB = 9000  # more than one 8192-token vocabulary chunk
B, S, H, K = 2, 4, 16, 8


def make(seed: int = 0):
    torch.manual_seed(seed)
    head = GemmaFusedOutputLinear(H, VOCAB, chunk_size=3, softcap=SOFTCAP)
    head.weight.data = torch.randn(VOCAB, H) * 2.0
    hidden = (torch.randn(B, S, H) * 2.0).requires_grad_()
    labels = torch.randint(0, VOCAB, (B, S))
    temperature = 0.5 + torch.rand(B, S)
    mask = torch.randint(0, VOCAB, (B, S, K))
    mask[..., 0] = labels  # the label is in its mask
    mask[..., -2:] = -1  # padding
    mask[0, 0] = -1  # empty mask: falls back to the full vocabulary
    mask[0, 1] = (labels[0, 1] + 1) % VOCAB  # label outside its mask: falls back too
    return head, hidden, labels, temperature, mask


def reference(head, hidden, labels, temperature, mask):
    logits = SOFTCAP * torch.tanh(hidden @ head.weight.t() / SOFTCAP) / temperature.unsqueeze(-1)
    full = torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if mask is None:
        return full
    in_mask = torch.where(mask >= 0, logits.gather(-1, mask.clamp_min(0)), torch.tensor(float("-inf")))
    replay = (mask >= 0).any(-1) & (mask == labels.unsqueeze(-1)).any(-1)
    replayed = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(in_mask, -1)
    return torch.where(replay, replayed, full)


def check(use_mask: bool):
    head, hidden, labels, temperature, mask = make()
    mask = mask if use_mask else None
    out = head(hidden, labels=labels, temperature=temperature, sampling_mask=mask)
    out["logprobs"].sum().backward()
    grad_hidden, grad_weight = hidden.grad.clone(), head.weight.grad.clone()

    hidden.grad, head.weight.grad = None, None
    expected = reference(head, hidden, labels, temperature, mask)
    assert torch.allclose(out["logprobs"].detach(), expected.detach(), rtol=1e-4, atol=1e-4)
    expected.sum().backward()
    assert torch.allclose(grad_hidden, hidden.grad, rtol=1e-3, atol=1e-3)
    assert torch.allclose(grad_weight, head.weight.grad, rtol=1e-3, atol=1e-3)


def test_logprobs_and_gradients_with_sampling_mask():
    check(use_mask=True)


def test_logprobs_and_gradients_without_sampling_mask():
    check(use_mask=False)


def test_mask_changes_replayed_positions_only():
    head, hidden, labels, temperature, mask = make()
    full = head(hidden, labels=labels, temperature=temperature)["logprobs"]
    masked = head(hidden, labels=labels, temperature=temperature, sampling_mask=mask)["logprobs"]
    assert torch.allclose(full[0, :2], masked[0, :2], rtol=1e-6, atol=1e-6)  # the two fallback positions
    # Renormalizing over a subset raises logprobs.
    assert (masked[0, 2:] >= full[0, 2:] - 1e-6).all() and (masked[1] > full[1]).all()


def test_loss_path_still_backpropagates():
    head, hidden, labels, _, _ = make()
    head(hidden, labels=labels)["loss"].backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
