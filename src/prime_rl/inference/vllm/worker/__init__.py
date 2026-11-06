from prime_rl.utils.olmo3_rope import patch_olmo3_rope

# Every weight-update worker imports this package before vLLM builds the model.
patch_olmo3_rope()
