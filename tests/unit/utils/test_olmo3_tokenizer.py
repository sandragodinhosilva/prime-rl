import json

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

from prime_rl.utils.olmo3_tokenizer import patch_olmo3_tokenizer

UNPATCHED = AutoTokenizer.__dict__["from_pretrained"]
# The Split rule in Olmo 3's tokenizer.json (cl100k's pattern). GPT-2's byte-level regex cuts "():\n" into "():" + "\n".
CL100K = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)"
    r"|\s+"
)
PROGRAM = 'def main():\n    n = int(input())\n    print(n * 2)\n\nif __name__ == "__main__":\n    main()\n'
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CHAT = [
    {"role": "system", "content": "You are an expert Python programmer."},
    {"role": "user", "content": f"Fix this ():\n```python\n{PROGRAM}```"},
]


def checkpoint(path, model_type: str) -> Tokenizer:
    """A small checkpoint laid out as Olmo 3's: a tokenizer.json with the cl100k Split rule, GPT2Tokenizer in
    tokenizer_config.json, `model_type` in config.json. Returns the tokenizer.json as `tokenizers` reads it."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(CL100K), behavior="removed", invert=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [PROGRAM, CHAT[1]["content"]] * 8,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=SPECIAL,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    path.mkdir()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "GPT2Tokenizer",
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "add_prefix_space": False,
                "chat_template": TEMPLATE,
            }
        )
    )
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return Tokenizer.from_file(str(path / "tokenizer.json"))


def ids(tok, text):
    return list(tok.encode(text, add_special_tokens=False))


def chat_ids(tok):
    return list(tok.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=True, return_dict=False))


@pytest.fixture(autouse=True)
def unpatched_auto_tokenizer(monkeypatch):
    """Each test starts from transformers' own AutoTokenizer.from_pretrained and leaves it as it found it."""
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", UNPATCHED)


def test_olmo3_checkpoint_encodes_as_its_tokenizer_json(tmp_path):
    raw = checkpoint(tmp_path / "olmo3", "olmo3")
    own = lambda text: raw.encode(text, add_special_tokens=False).ids  # noqa: E731
    before = AutoTokenizer.from_pretrained(tmp_path / "olmo3")
    assert type(before).__name__ == "GPT2Tokenizer" and ids(before, PROGRAM) != own(PROGRAM), "the pin's split, the bug"
    patch_olmo3_tokenizer()
    patch_olmo3_tokenizer()
    after = AutoTokenizer.from_pretrained(tmp_path / "olmo3")
    assert type(after).__name__ == "TokenizersBackend"
    assert ids(after, PROGRAM) == own(PROGRAM)
    text = after.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    assert chat_ids(after) == own(text)
    assert text == before.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "all_special_ids", "chat_template"):
        assert getattr(after, name) == getattr(before, name), name
    assert len(after) == len(before) and after.decode(own(PROGRAM)) == PROGRAM


def test_other_model_types_keep_their_tokenizer(tmp_path):
    checkpoint(tmp_path / "granite", "granite")  # GPT2Tokenizer on the hub, as Granite 4.2
    before = AutoTokenizer.from_pretrained(tmp_path / "granite")
    patch_olmo3_tokenizer()
    after = AutoTokenizer.from_pretrained(tmp_path / "granite")
    assert type(after) is type(before)
    assert ids(after, PROGRAM) == ids(before, PROGRAM) and chat_ids(after) == chat_ids(before)


def test_every_policy_tokenizer_loader_applies_it(tmp_path):
    """prime-rl's setup_tokenizer (orchestrator, SFT), the env server's worker setup (renderers' load_tokenizer, training
    prompts) and the vLLM general plugin (the inference server's tokenizer, eval prompts) each patch AutoTokenizer."""
    from prime_rl.configs.trainer import TokenizerConfig
    from prime_rl.entrypoints.env_server import setup_worker
    from prime_rl.inference.patches import apply_shared_vllm_patches
    from prime_rl.trainer.model import setup_tokenizer

    raw = checkpoint(tmp_path / "olmo3", "olmo3")
    assert (
        ids(setup_tokenizer(TokenizerConfig(name=str(tmp_path / "olmo3"))), PROGRAM)
        == raw.encode(PROGRAM, add_special_tokens=False).ids
    )
    for apply in (lambda: setup_worker("info", False, []), apply_shared_vllm_patches):
        AutoTokenizer.from_pretrained = UNPATCHED
        apply()
        assert type(AutoTokenizer.from_pretrained(tmp_path / "olmo3")).__name__ == "TokenizersBackend", apply

    from vllm.tokenizers.registry import get_tokenizer

    assert ids(get_tokenizer(str(tmp_path / "olmo3")), PROGRAM) == raw.encode(PROGRAM, add_special_tokens=False).ids
