import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import AppState, CheckpointManager, PendingSave, PinnedStager, Progress
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


def test_keep_last_ignores_a_step_cut_off_mid_write(tmp_path):
    ckpt_dir = get_ckpt_dir(tmp_path)
    # Steps 1 and 2 finished writing; the run crashed while writing step 3 and resumes from step 2
    for step in (1, 2, 3):
        (get_step_path(ckpt_dir, step) / "trainer").mkdir(parents=True)
    for step in (1, 2):
        (get_step_path(ckpt_dir, step) / "trainer" / ".metadata").touch()

    manager = CheckpointManager(tmp_path, CheckpointConfig(keep_last=1))
    manager.maybe_clean()

    assert manager.ckpt_steps == [2]
    assert not get_step_path(ckpt_dir, 1).exists()
    assert get_step_path(ckpt_dir, 2).exists()


def pending_with(step: int, write: Future) -> PendingSave:
    return PendingSave(step=step, staging=None, write=write, start_time=0.0)


def test_a_finished_async_write_counts_only_once_waited_for(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointConfig(async_save=True, keep_last=1))
    write: Future = Future()
    manager.pending = pending_with(5, write)
    manager.wait_for_staging()  # still writing: the step is not registered
    assert manager.ckpt_steps == [] and manager.pending is not None
    write.set_result(None)
    manager.wait_for_staging()  # the fence registers a write that finished meanwhile
    assert manager.ckpt_steps == [5] and manager.pending is None
    manager.wait()  # nothing pending: a no-op
    assert manager.ckpt_steps == [5]


def test_a_failed_async_write_raises_at_the_next_wait(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointConfig(async_save=True))
    write: Future = Future()
    write.set_exception(OSError("disk full"))
    manager.pending = pending_with(5, write)
    with pytest.raises(OSError, match="disk full"):
        manager.wait()
    assert manager.ckpt_steps == []  # a failed write is never counted


def test_default_is_synchronous():
    assert CheckpointConfig().async_save is False


@pytest.mark.gpu
def test_async_save_matches_sync_save_and_resumes(tmp_path) -> None:
    """The async write is byte-for-byte the state a synchronous save writes (loaded back, compared), is complete only once
    `.metadata` exists, and a write that only starts after save() returned still gets the staged state."""
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29516", world_size=1, rank=0)
    try:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(2):
            model(torch.randn(8, 16, device="cuda")).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad()

        sync = CheckpointManager(tmp_path / "sync", CheckpointConfig())
        sync.save(1, model, [optimizer], None, Progress(step=1))

        manager = CheckpointManager(tmp_path / "async", CheckpointConfig(async_save=True))
        manager.stager = PinnedStager()
        manager.write_group = dist.new_group(backend="gloo")
        manager.writer = ThreadPoolExecutor(max_workers=1)
        release = threading.Event()
        manager.writer.submit(release.wait)  # hold the writer until save() has returned
        manager.save(1, model, [optimizer], None, Progress(step=1))
        assert not (manager.get_ckpt_path(1) / ".metadata").exists()
        # the optimizer step the fence protects: the staged copy must not see it
        manager.wait_for_staging()
        model(torch.randn(8, 16, device="cuda")).square().mean().backward()
        optimizer.step()
        release.set()
        manager.wait()
        assert (manager.get_ckpt_path(1) / ".metadata").exists() and manager.ckpt_steps == [1]

        def load(manager_dir):
            fresh = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)).cuda()
            opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2)
            progress = Progress()
            dcp_load({"app": AppState(fresh, [opt], None, progress)}, checkpoint_id=manager_dir)
            return fresh, opt, progress

        (m_sync, o_sync, p_sync), (m_async, o_async, p_async) = (
            load(sync.get_ckpt_path(1)),
            load(manager.get_ckpt_path(1)),
        )
        assert p_sync == p_async
        for a, b in zip(m_sync.parameters(), m_async.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(o_sync.state.values(), o_async.state.values()):
            assert all(torch.equal(a[k], b[k]) for k in a)
    finally:
        dist.destroy_process_group()
