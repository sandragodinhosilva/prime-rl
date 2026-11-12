"""GET /weight_version and the workers' weight version: what a client keys a vLLM cache_salt on when the policy's own server
also serves a judge whose verdicts must not reuse prefix-cache blocks computed under older weights (spb self live)."""
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from prime_rl.inference.vllm import server
from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker
from prime_rl.inference.vllm.worker.weight_transfer import step_of


def test_step_of_reads_the_update_directory_name():
    assert step_of("/run/broadcasts/step_120") == 120
    assert step_of("/run/weights/step_7/") == 7
    assert step_of(None) is None
    assert step_of("/run/broadcasts/latest") is None
    assert step_of("/run/broadcasts/step_x") is None


def test_workers_report_version_zero_before_any_update():
    for cls in (NCCLWeightUpdateWorker, FileSystemWeightUpdateWorker):
        worker = cls()
        assert worker.weight_version() == 0
        worker._weight_version = 42
        assert worker.weight_version() == 42


def _app(paused: bool, versions: list):
    app = FastAPI()
    app.include_router(server.router)
    engine = AsyncMock()
    engine.is_paused.return_value = paused
    engine.collective_rpc.return_value = versions
    app.state.engine_client = engine
    return TestClient(app), engine


def test_weight_version_route():
    client, engine = _app(False, [120, 120, 120, 120])
    response = client.get("/weight_version")
    assert response.status_code == 200 and response.json() == {"version": 120}
    engine.collective_rpc.assert_awaited_once_with("weight_version")
    assert _app(True, [120])[0].get("/weight_version").status_code == 503  # paused: an update may be under way
    assert _app(False, [120, 119])[0].get("/weight_version").status_code == 503  # engines mid-update
    assert _app(False, [None, None])[0].get("/weight_version").status_code == 409  # the update's dir carried no step
