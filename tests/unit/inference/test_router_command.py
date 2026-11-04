from jinja2 import Environment, FileSystemLoader

from prime_rl.configs.inference import InferenceConfig, VllmRouterConfig
from prime_rl.entrypoints.inference import router_command

TEMPLATES = "src/prime_rl/templates"


def flag(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_router_health_flags_default_to_the_routers_own():
    cmd = router_command(InferenceConfig.model_validate({}))
    assert flag(cmd, "--health-check-timeout-secs") is None
    assert flag(cmd, "--health-failure-threshold") is None


def test_router_health_flags_are_passed_when_set():
    config = InferenceConfig.model_validate(
        {"router": {"type": "vllm-router", "health_check_timeout_secs": 60, "health_failure_threshold": 10}}
    )
    cmd = router_command(config)
    assert flag(cmd, "--health-check-timeout-secs") == "60"
    assert flag(cmd, "--health-failure-threshold") == "10"


def test_slurm_router_template_passes_health_flags_only_when_set():
    template = Environment(loader=FileSystemLoader(TEMPLATES)).get_template("_launch_router.sh.j2")
    default = template.render(router=VllmRouterConfig())
    assert "--health-check-timeout-secs" not in default and "--health-failure-threshold" not in default
    set_ = template.render(router=VllmRouterConfig(health_check_timeout_secs=60, health_failure_threshold=10))
    assert "--health-check-timeout-secs 60" in set_ and "--health-failure-threshold 10" in set_


def test_router_circuit_breaker_is_kept_unless_disabled():
    assert "--disable-circuit-breaker" not in router_command(InferenceConfig.model_validate({}))
    config = InferenceConfig.model_validate({"router": {"type": "vllm-router", "disable_circuit_breaker": True}})
    assert "--disable-circuit-breaker" in router_command(config)
    template = Environment(loader=FileSystemLoader(TEMPLATES)).get_template("_launch_router.sh.j2")
    assert "--disable-circuit-breaker" in template.render(router=VllmRouterConfig(disable_circuit_breaker=True))
    assert "--disable-circuit-breaker" not in template.render(router=VllmRouterConfig())
