"""Olmo 3 text encoded with the checkpoint's own tokenizer.json, under the pinned transformers.

Olmo 3 checkpoints name `GPT2Tokenizer` in tokenizer_config.json. transformers 5.6.2's AutoTokenizer prefers that hub class to
the `TokenizersBackend` it registers for model type `olmo3`, and `GPT2Tokenizer.__init__` keeps only the vocab and merges of
tokenizer.json: it rebuilds the pre-tokenizer as GPT-2's byte-level regex and drops the checkpoint's `Split` rule (the cl100k
pattern). Text is then cut into a split the model never writes, e.g. "():\\n" becomes "():" + "\\n" (an 8-line program: 57 tokens
against 39). transformers 5.15 loads the registered TokenizersBackend, which reads tokenizer.json whole. Every AutoTokenizer call
reads it the same way: prime-rl's `setup_tokenizer`, the renderers' `load_tokenizer` in the env servers, and vLLM's tokenizer in
the inference server. Here AutoTokenizer gets `tokenizer_type=<model type>` for these model types, transformers' own switch to the
registered class. Other model types are untouched.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils.hub import cached_file

# Model types whose checkpoints name a tokenizer class that rebuilds the pre-tokenizer, while transformers registers
# TokenizersBackend (tokenizer.json as is) for them.
TOKENIZER_JSON_MODEL_TYPES = frozenset({"olmo3"})

_HUB_KWARGS = ("cache_dir", "revision", "token", "local_files_only", "subfolder")


def config_model_type(name_or_path, kwargs: dict) -> str | None:
    """The model type AutoTokenizer resolves the tokenizer class with: a passed `config`'s, else config.json's (None when absent)."""
    if kwargs.get("config") is not None:
        return getattr(kwargs["config"], "model_type", None)
    path = Path(name_or_path) / (kwargs.get("subfolder") or "") / "config.json"
    if not path.is_file():
        found = cached_file(
            name_or_path,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **{k: kwargs[k] for k in _HUB_KWARGS if kwargs.get(k) is not None},
        )
        if found is None:
            return None
        path = Path(found)
    return json.loads(path.read_text()).get("model_type")


def patch_olmo3_tokenizer() -> None:
    """AutoTokenizer.from_pretrained loads TOKENIZER_JSON_MODEL_TYPES with their registered TokenizersBackend. Idempotent."""
    original = AutoTokenizer.__dict__["from_pretrained"].__func__
    if getattr(original, "_prime_rl_olmo3_tokenizer", False):
        return

    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if kwargs.get("tokenizer_type") is None and not kwargs.get("gguf_file"):
            model_type = config_model_type(pretrained_model_name_or_path, kwargs)
            if model_type in TOKENIZER_JSON_MODEL_TYPES:
                kwargs["tokenizer_type"] = model_type
        return original(cls, pretrained_model_name_or_path, *inputs, **kwargs)

    from_pretrained._prime_rl_olmo3_tokenizer = True
    from_pretrained.__doc__ = original.__doc__
    AutoTokenizer.from_pretrained = classmethod(from_pretrained)
