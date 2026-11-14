import json
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from prime_rl.utils.weights import (
    HANDSHAKE_DIR,
    name_global_shards,
    save_file_releasing_gil,
    write_export_files,
    write_rank_shards,
)


def _tensors() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    tensors = {
        "model.layers.1.mlp.weight": torch.randn(64, 32, generator=gen).to(torch.bfloat16),
        "model.layers.0.mlp.weight": torch.randn(64, 32, generator=gen).to(torch.bfloat16),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(8, generator=gen),
        "model.norm.weight": torch.randn(32, generator=gen).to(torch.float16),
        "scalar": torch.tensor(3.5, dtype=torch.float64),
        "empty": torch.zeros(0, 4, dtype=torch.bfloat16),
        "ids": torch.arange(5, dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
        "fp8": torch.randn(4, 4, generator=gen).to(torch.float8_e4m3fn),
        "u8": torch.arange(7, dtype=torch.uint8),
        "i32": torch.arange(3, dtype=torch.int32),
    }
    # an unstacked expert: a contiguous view into a stacked tensor, as convert_to_hf produces
    stacked = torch.randn(4, 16, 8, generator=gen).to(torch.bfloat16)
    tensors["model.layers.0.mlp.experts.2.down_proj.weight"] = stacked[2]
    return tensors


@pytest.mark.parametrize("metadata", [{"format": "pt"}, None])
def test_gil_free_writer_matches_save_file_byte_for_byte(tmp_path: Path, metadata):
    tensors = _tensors()
    save_file(tensors, tmp_path / "reference.safetensors", metadata=metadata)
    save_file_releasing_gil(tensors, tmp_path / "ours.safetensors", metadata=metadata, chunk_bytes=100)
    assert (tmp_path / "ours.safetensors").read_bytes() == (tmp_path / "reference.safetensors").read_bytes()


def test_gil_free_writer_falls_back_outside_its_layout(tmp_path: Path):
    tensors = {"t": torch.randn(4, 6)[:, ::2], "u": torch.randn(3)}  # non-contiguous
    save_file_releasing_gil({"u": tensors["u"]}, tmp_path / "a.safetensors")
    with pytest.raises(ValueError, match="non contiguous"):  # what save_file itself says
        save_file_releasing_gil(tensors, tmp_path / "b.safetensors")


def test_global_numbering_is_rank_order_then_write_order():
    all_shards = [
        [
            ("tmp-rank0-00001-of-00002.safetensors", {"a": 10, "b": 5}),
            ("tmp-rank0-00002-of-00002.safetensors", {"c": 1}),
        ],
        [],
        [("tmp-rank2.safetensors", {"d": 7})],
    ]
    renames, weight_map, total_size = name_global_shards(all_shards)
    assert renames == [
        (0, "tmp-rank0-00001-of-00002.safetensors", "model-00001-of-00003.safetensors"),
        (0, "tmp-rank0-00002-of-00002.safetensors", "model-00002-of-00003.safetensors"),
        (2, "tmp-rank2.safetensors", "model-00003-of-00003.safetensors"),
    ]
    assert weight_map == {
        "a": "model-00001-of-00003.safetensors",
        "b": "model-00001-of-00003.safetensors",
        "c": "model-00002-of-00003.safetensors",
        "d": "model-00003-of-00003.safetensors",
    }
    assert total_size == 23
    assert name_global_shards([[("tmp-rank0.safetensors", {"a": 1})]])[0] == [
        (0, "tmp-rank0.safetensors", "model.safetensors")
    ]


def _slices(world_size: int) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(1)
    slices = []
    for rank in range(world_size):
        slices.append(
            {
                f"model.layers.{rank}.{name}": torch.randn(256, 64, generator=gen).to(torch.bfloat16)
                for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight")
            }
        )
    return slices


def _sync_reference(slices: list[dict[str, torch.Tensor]], save_dir: Path, base_dir: Path) -> None:
    """What the synchronous path writes, from the same slices: each rank's shards, then the
    shared numbering, the index and the base files (save_state_dict_parallel and the tail
    of export_policy_weights without torch.distributed)."""
    import prime_rl.utils.weights as weights

    save_dir.mkdir()
    all_shards = [write_rank_shards(dict(sl), save_dir, rank) for rank, sl in enumerate(slices)]
    renames, weight_map, total_size = name_global_shards(all_shards)
    for _, tmp_name, final_name in renames:
        (save_dir / tmp_name).rename(save_dir / final_name)
    if len(renames) > 1:
        weights.write_weights_index(save_dir, weight_map, total_size)
    weights._copy_base_files(base_dir, save_dir)
    (save_dir / ".complete").touch()


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type": "toy"}')
    (base / "tokenizer.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"not copied")
    (base / SAFE_WEIGHTS_INDEX_NAME).write_text("{}")
    return base


def _run_ranks(slices, save_dir: Path, base_dir: Path, **kwargs) -> list[BaseException | None]:
    errors: list[BaseException | None] = [None] * len(slices)

    def run(rank: int) -> None:
        try:
            write_export_files(dict(slices[rank]), save_dir, base_dir, rank, len(slices), poll=0.01, **kwargs)
        except BaseException as error:
            errors[rank] = error

    # ranks start in reverse, so rank 0 is the last to publish its shard list
    threads = [threading.Thread(target=run, args=(rank,)) for rank in reversed(range(len(slices)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return errors


def test_async_handshake_writes_what_the_sync_path_writes(tmp_path: Path, base_dir: Path, monkeypatch):
    import prime_rl.utils.weights as weights

    # small shards so every rank writes several and the numbering crosses ranks
    original = weights.split_torch_state_dict_into_shards
    monkeypatch.setattr(
        weights,
        "split_torch_state_dict_into_shards",
        lambda state_dict, filename_pattern: original(
            state_dict, filename_pattern=filename_pattern, max_shard_size=70_000
        ),
    )
    slices = _slices(world_size=3)
    _sync_reference(slices, tmp_path / "sync", base_dir)

    save_dir = tmp_path / "async"
    save_dir.mkdir()
    (save_dir / HANDSHAKE_DIR).mkdir()
    assert _run_ranks(slices, save_dir, base_dir) == [None, None, None]

    sync_files = sorted(p.name for p in (tmp_path / "sync").iterdir())
    async_files = sorted(p.name for p in save_dir.iterdir())
    assert async_files == sync_files  # handshake gone, nothing else left
    assert sum(name.endswith(".safetensors") for name in async_files) == 6
    assert "model.safetensors" not in async_files  # the base model's weights are not copied
    for name in sync_files:
        assert (save_dir / name).read_bytes() == (tmp_path / "sync" / name).read_bytes(), name
    index = json.loads((save_dir / SAFE_WEIGHTS_INDEX_NAME).read_text())
    loaded = {}
    for shard in set(index["weight_map"].values()):
        loaded.update(load_file(save_dir / shard))
    expected = {k: v for sl in slices for k, v in sl.items()}
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(loaded[key], value)
    assert index["metadata"]["total_size"] == sum(v.nbytes for v in expected.values())


def test_async_handshake_one_rank(tmp_path: Path, base_dir: Path):
    slices = _slices(world_size=1)
    save_dir = tmp_path / "async"
    save_dir.mkdir()
    (save_dir / HANDSHAKE_DIR).mkdir()
    assert _run_ranks(slices, save_dir, base_dir) == [None]
    assert sorted(p.name for p in save_dir.iterdir()) == [
        ".complete",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]


def test_a_failing_rank_stops_the_others_and_leaves_no_marker(tmp_path: Path, base_dir: Path, monkeypatch):
    import prime_rl.utils.weights as weights

    original = weights.save_file_releasing_gil

    def fail_on_rank1(tensors, path, metadata=None):
        if "tmp-rank1" in path.name:
            raise OSError("disk full")
        original(tensors, path, metadata=metadata)

    monkeypatch.setattr(weights, "save_file_releasing_gil", fail_on_rank1)
    slices = _slices(world_size=3)
    save_dir = tmp_path / "async"
    save_dir.mkdir()
    (save_dir / HANDSHAKE_DIR).mkdir()
    errors = _run_ranks(slices, save_dir, base_dir)
    assert isinstance(errors[1], OSError)
    assert isinstance(errors[0], RuntimeError) and isinstance(errors[2], RuntimeError)
    assert not (save_dir / ".complete").exists()


def test_a_missing_rank_times_out(tmp_path: Path, base_dir: Path):
    slices = _slices(world_size=2)
    save_dir = tmp_path / "async"
    save_dir.mkdir()
    (save_dir / HANDSHAKE_DIR).mkdir()
    with pytest.raises(TimeoutError):
        write_export_files(dict(slices[0]), save_dir, base_dir, 0, 2, timeout=0.2, poll=0.01)
    assert not (save_dir / ".complete").exists()


def test_wait_for_weight_export_joins_and_reraises(monkeypatch):
    import prime_rl.utils.weights as weights

    done = threading.Event()
    pending = weights._PendingExport(Path("/nonexistent/step_10"))
    pending.thread = threading.Thread(target=lambda: (done.wait(5), None))
    pending.thread.start()
    monkeypatch.setattr(weights, "_pending_export", pending)
    done.set()
    assert weights.wait_for_weight_export() >= 0
    assert weights._pending_export is None
    assert weights.wait_for_weight_export() == 0.0  # nothing in flight

    failed = weights._PendingExport(Path("/nonexistent/step_20"))
    failed.error = OSError("disk full")
    failed.thread = threading.Thread(target=lambda: None)
    failed.thread.start()
    monkeypatch.setattr(weights, "_pending_export", failed)
    with pytest.raises(RuntimeError, match="step_20"):
        weights.wait_for_weight_export()


def test_a_reexport_drops_the_stale_marker_before_writing(tmp_path: Path, base_dir: Path, monkeypatch):
    """A resumed run exports a step again into a directory whose .complete vouches for the old files:
    the marker goes before the first new byte is written and comes back only at the end."""
    import socket

    import torch.distributed as dist

    import prime_rl.utils.weights as weights

    save_dir = tmp_path / "step_3"
    save_dir.mkdir()
    (save_dir / "model.safetensors").write_bytes(b"old weights")
    (save_dir / ".complete").touch()
    marker_seen: list[bool] = []
    original = weights.write_rank_shards

    def spy(*args, **kwargs):
        marker_seen.append((save_dir / ".complete").exists())
        return original(*args, **kwargs)

    monkeypatch.setattr(weights, "write_rank_shards", spy)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")
    try:
        new = _slices(world_size=1)[0]
        weights._start_async_export(dict(new), save_dir, base_dir)
        weights.wait_for_weight_export()
    finally:
        dist.destroy_process_group()
    assert marker_seen == [False]
    assert (save_dir / ".complete").exists() and not (save_dir / HANDSHAKE_DIR).exists()
    loaded = load_file(save_dir / "model.safetensors")
    assert loaded.keys() == new.keys() and all(torch.equal(loaded[k], v) for k, v in new.items())
