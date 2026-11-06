import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

from prime_rl.utils.olmo3_rope import patch_olmo3_rope, plain_rope

YARN = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 64, "attention_factor": 1.2,
        "beta_fast": 32.0, "beta_slow": 1.0, "rope_theta": 500000}


def rotaries_reaching_attention(position_ids):
    config = Olmo3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=2, sliding_window=8, max_position_embeddings=512, rope_parameters=YARN,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = Olmo3ForCausalLM(config).eval()
    seen = {}

    def record(index):
        def hook(module, args, kwargs):
            seen[index] = kwargs["position_embeddings"]
        return hook

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_pre_hook(record(i), with_kwargs=True)
    with torch.no_grad():
        model(input_ids=torch.ones_like(position_ids), position_ids=position_ids)
    return config, seen


def test_sliding_layers_get_plain_rope_and_full_layers_keep_yarn():
    patch_olmo3_rope()
    position_ids = torch.arange(16)[None]
    config, seen = rotaries_reaching_attention(position_ids)
    cos, sin = plain_rope(config, position_ids, torch.float32)
    for i, layer_type in enumerate(config.layer_types):
        if layer_type == "sliding_attention":
            assert torch.equal(seen[i][0], cos) and torch.equal(seen[i][1], sin), i
        else:
            assert seen[i][0].abs().max() > 1.1, "full-attention layers keep YaRN's attention factor"


def test_plain_rope_restarts_with_packed_samples():
    patch_olmo3_rope()
    position_ids = torch.cat((torch.arange(6), torch.arange(10)))[None]
    config, seen = rotaries_reaching_attention(position_ids)
    cos = seen[0][0]
    assert torch.equal(cos[:, 6:12], cos[:, :6]), "the second sample starts again at position 0"
