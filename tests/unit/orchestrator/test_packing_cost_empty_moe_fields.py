from prime_rl.orchestrator.batch import _packing_cost_coeffs


def test_packing_cost_of_a_dense_config_with_empty_moe_fields():
    class Dense:
        hidden_size, vocab_size, num_hidden_layers, num_attention_heads = 8, 16, 2, 2
        intermediate_size = 32
        num_experts = None
        moe_intermediate_size = None

    linear, quadratic = _packing_cost_coeffs(Dense())
    assert linear > 0 and quadratic > 0
