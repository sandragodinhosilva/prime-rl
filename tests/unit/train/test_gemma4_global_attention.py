"""packed_sdpa_forward against a direct softmax-attention computation, one packed sequence at a time."""

from types import SimpleNamespace

import torch

from prime_rl.trainer.models.gemma4_global_attention import packed_sdpa_forward

BATCH, HEADS, KV_HEADS, SEQ, DIM = 2, 8, 2, 12, 16


def direct(query, key, value, position_ids, scaling):
    """Per row and per sequence, softmax over the causal past."""
    groups = HEADS // KV_HEADS
    out = torch.zeros(BATCH, SEQ, HEADS, DIM)
    for row in range(BATCH):
        starts = (position_ids[row] == 0).nonzero().view(-1).tolist()
        for a, b in zip(starts, starts[1:] + [SEQ]):
            q = query[row, :, a:b]
            k = key[row, :, a:b].repeat_interleave(groups, 0)
            v = value[row, :, a:b].repeat_interleave(groups, 0)
            scores = (q @ k.transpose(-1, -2)) * scaling
            mask = torch.ones(b - a, b - a, dtype=torch.bool).tril()
            attended = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ v
            out[row, a:b] = attended.transpose(0, 1)
    return out


def make():
    torch.manual_seed(0)
    query = torch.randn(BATCH, HEADS, SEQ, DIM).requires_grad_()
    key = torch.randn(BATCH, KV_HEADS, SEQ, DIM).requires_grad_()
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1]])
    return query, key, position_ids


def test_matches_direct_computation_with_values_equal_to_keys():
    query, key, position_ids = make()
    module = SimpleNamespace(num_key_value_groups=HEADS // KV_HEADS, is_causal=True)
    out, weights = packed_sdpa_forward(module, query, key, key, None, scaling=1.0, position_ids=position_ids)
    expected = direct(query, key, key, position_ids, 1.0)
    assert weights is None and out.shape == (BATCH, SEQ, HEADS, DIM)
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5)

    weight = torch.randn(out.shape)
    (out * weight).sum().backward()
    grad_query, grad_key = query.grad.clone(), key.grad.clone()
    query.grad, key.grad = None, None
    (expected * weight).sum().backward()
    assert torch.allclose(grad_query, query.grad, rtol=1e-3, atol=1e-4)
    assert torch.allclose(grad_key, key.grad, rtol=1e-3, atol=1e-4)


def test_no_attention_across_packed_sequences():
    query, key, position_ids = make()
    module = SimpleNamespace(num_key_value_groups=HEADS // KV_HEADS, is_causal=True)
    out, _ = packed_sdpa_forward(module, query, key, key, None, scaling=1.0, position_ids=position_ids)
    # A sequence's first token attends only to itself, so its output is its own value vector.
    expected_first = key[0, :, 4].repeat_interleave(HEADS // KV_HEADS, 0)
    assert torch.allclose(out[0, 4], expected_first, rtol=1e-4, atol=1e-5)
