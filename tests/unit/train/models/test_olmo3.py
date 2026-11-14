"""PrimeRL Olmo 3 against transformers' Olmo3ForCausalLM WITH the spb RoPE fix (prime_rl.utils.olmo3_rope, a592ade85: plain RoPE
on the sliding layers, YaRN on the full ones), under the pinned transformers 5.6.2, which reads Olmo 3's rope as one flat YaRN
dict. Runs on CPU: the flash-attention kernel is replaced by a causal sliding-window SDPA over packed documents, so the test
covers everything but the kernel (layer order, norms, rotary per layer type, names)."""

import copy

import torch
import torch.nn.functional as F
from transformers import Olmo3Config
from transformers import Olmo3ForCausalLM as HFOlmo3ForCausalLM

from prime_rl.trainer.models.olmo3 import Olmo3ForCausalLM
from prime_rl.trainer.models.olmo3.modeling_olmo3 import Olmo3Attention
from prime_rl.utils.olmo3_rope import patch_olmo3_rope

YARN = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 64, "attention_factor": 1.2,
        "beta_fast": 32.0, "beta_slow": 1.0, "rope_theta": 500000}


def _sdpa_varlen(self, q, k, v, cu_seqlens, max_seqlen):
    out = torch.empty_like(q)
    window = getattr(self, "sliding_window", None)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        i = torch.arange(end - start)
        mask = i[:, None] >= i[None, :]
        if window is not None:
            mask &= (i[:, None] - i[None, :]) < window
        qh, kh, vh = (t[start:end].transpose(0, 1) for t in (q, k, v))
        rep = qh.shape[0] // kh.shape[0]
        kh, vh = kh.repeat_interleave(rep, 0), vh.repeat_interleave(rep, 0)
        out[start:end] = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=self.scaling).transpose(0, 1)
    return out


def _models():
    patch_olmo3_rope()
    torch.manual_seed(0)
    config = Olmo3Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, sliding_window=6, max_position_embeddings=512, rope_parameters=YARN,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2, rms_norm_eps=1e-6)
    config._attn_implementation = "sdpa"
    hf = HFOlmo3ForCausalLM(copy.deepcopy(config)).float().eval()
    config._attn_implementation = "flash_attention_2"
    prime = Olmo3ForCausalLM(config).float().eval()
    prime.init_buffers_post_meta()
    missing, unexpected = prime.load_state_dict(hf.state_dict(), strict=False)
    assert not unexpected and not [m for m in missing if "inv_freq" not in m], (missing, unexpected)
    return config, hf, prime


def test_hidden_states_match_transformers_with_rope_fix(monkeypatch):
    monkeypatch.setattr(Olmo3Attention, "_compute_attention", _sdpa_varlen)
    config, hf, prime = _models()
    assert config.layer_types.count("sliding_attention") == 3
    ids = torch.randint(3, 97, (1, 20))
    with torch.no_grad():
        ref = hf.model(input_ids=ids).last_hidden_state
        out = prime.model(input_ids=ids, seq_lens=torch.tensor([20])).last_hidden_state
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_packed_documents_restart_positions(monkeypatch):
    monkeypatch.setattr(Olmo3Attention, "_compute_attention", _sdpa_varlen)
    _, hf, prime = _models()
    a, b = torch.randint(3, 97, (1, 9)), torch.randint(3, 97, (1, 11))
    with torch.no_grad():
        ref = torch.cat([hf.model(input_ids=x).last_hidden_state for x in (a, b)], dim=1)
        pos = torch.cat([torch.arange(9), torch.arange(11)])[None]
        out = prime.model(input_ids=torch.cat([a, b], 1), position_ids=pos, seq_lens=torch.tensor([9, 11])).last_hidden_state
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_flat_rope_maps_to_yarn_full_and_plain_sliding():
    _, _, prime = _models()
    rot = prime.model.rotary_emb
    assert not rot.keyed and rot.rope_type == {"full_attention": "yarn", "sliding_attention": "default"}
    assert rot.full_attention_attention_scaling == 1.2 and rot.sliding_attention_attention_scaling == 1.0
