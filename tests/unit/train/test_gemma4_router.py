from functools import partial

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter

from prime_rl.trainer.models.gemma4_router import patch_gemma4_router


def save_topk(_context, operation, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if operation.name() == "aten::topk" else CheckpointPolicy.PREFER_RECOMPUTE


@pytest.fixture
def router():
    original = Gemma4TextRouter.forward
    torch.manual_seed(0)
    yield Gemma4TextRouter(Gemma4TextConfig(hidden_size=8, num_experts=4, top_k_experts=2)), original
    Gemma4TextRouter.forward = original


def backward_through_checkpoint(module, hidden):
    context = partial(create_selective_checkpoint_contexts, save_topk)
    weights = checkpoint(lambda x: module(x)[1], hidden, use_reentrant=False, context_fn=context)
    weights.sum().backward()


def test_same_outputs_as_the_original(router):
    module, _ = router
    hidden = torch.randn(6, 8)
    expected = module(hidden)
    patch_gemma4_router()
    for got, want in zip(module(hidden), expected):
        assert torch.equal(got, want)


def test_original_fails_and_patched_backpropagates_through_selective_checkpointing(router):
    module, _ = router
    with pytest.raises(RuntimeError, match="has been mutated"):
        backward_through_checkpoint(module, torch.randn(6, 8, requires_grad=True))
    patch_gemma4_router()
    hidden = torch.randn(6, 8, requires_grad=True)
    backward_through_checkpoint(module, hidden)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
