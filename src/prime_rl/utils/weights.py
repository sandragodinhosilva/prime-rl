import json
import os
import re
import shutil
import struct
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, cast

import torch
import torch.distributed as dist
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


def load_state_dict_keys(save_dir: Path) -> list[str]:
    """Load only the key names from safetensor files without reading tensor data."""
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory as safetensors."""
    logger = get_logger()
    weights_name = ADAPTER_SAFE_WEIGHTS_NAME if adapter else SAFE_WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        # Save weights (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4252)
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]
            save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = save_dir / SAFE_WEIGHTS_INDEX_NAME
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})


def convert_state_dict_to_hf(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a (possibly rank-partial) training-format state dict to HF hub format.

    Format detection uses the model's full key set, so a partial dict (one rank's
    slice) converts the same way as the full state dict would.
    """
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    full_keys = dict.fromkeys(resolve_fqn(model, key) for key in model.state_dict().keys())
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(full_keys):
        # PrimeRL custom model holding weights in prime format: apply the model's
        # declarative prime->HF conversion chain (renames, expert stack/unstack).
        return model.convert_to_hf(state_dict)
    else:
        # Plain transformers model: undo the key renames transformers applied when
        # it loaded the HF checkpoint.
        from transformers.core_model_loading import revert_weight_conversion

        return revert_weight_conversion(model, state_dict)


def resolve_fqn(model: nn.Module, key: str) -> str:
    """Resolve a state-dict key to the parameter's canonical fully-qualified name.

    Strips wrapper prefixes that training composes onto module paths, e.g. with
    ``torch.compile`` the key ``model._orig_mod.layers.0.self_attn.q_proj.weight``
    resolves to ``model.layers.0.self_attn.q_proj.weight`` — the name the tensor
    has in the HF checkpoint.
    """
    fqns = get_fqns(model, key)
    assert len(fqns) == 1
    return next(iter(fqns))


_LAYER_RE = re.compile(r"^(.*?\blayers\.\d+)\.")


def partition_weights(state_dict: dict[str, Tensor], world_size: int, dtype: torch.dtype) -> dict[str, int]:
    """Assign each state-dict key an owner rank, balanced by byte size.

    All keys of one decoder layer share an owner, so per-rank prime->HF conversion
    (which stacks/unstacks tensors within a layer) always sees complete layers.
    """
    key_to_unit: dict[str, str] = {}
    unit_bytes: dict[str, int] = {}
    for key, value in state_dict.items():
        match = _LAYER_RE.match(key)
        unit = match.group(1) if match else ""
        key_to_unit[key] = unit
        itemsize = dtype.itemsize if isinstance(value, DTensor) else value.element_size()
        unit_bytes[unit] = unit_bytes.get(unit, 0) + value.numel() * itemsize
    loads = [0] * world_size
    unit_owner: dict[str, int] = {}
    for unit in sorted(unit_bytes, key=lambda unit: (-unit_bytes[unit], unit)):
        owner = min(range(world_size), key=lambda rank: loads[rank])
        unit_owner[unit] = owner
        loads[owner] += unit_bytes[unit]
    return {key: unit_owner[unit] for key, unit in key_to_unit.items()}


def resolve_wire_dtype(keep_in_fp32: Callable[[str], bool] | None, key: str, default: torch.dtype) -> torch.dtype:
    """The dtype a tensor must travel to the inference engine in.

    TODO: NIXL keeps its own inline copy of this rule; unify the handling.
    """
    return torch.float32 if keep_in_fp32 is not None and keep_in_fp32(key) else default


def gather_weights_parallel(
    model: nn.Module, dtype: torch.dtype = torch.bfloat16, copy: bool = False
) -> dict[str, Tensor]:
    """Gather distributed weights cooperatively, each rank keeping a slice on CPU.

    Every rank participates in the per-tensor all-gathers (a ``full_tensor`` call is
    collective and materializes the full tensor on all ranks anyway), but instead of
    only the master keeping everything, each rank copies its owned slice to CPU —
    so the D2H traffic and the shard writes are split across ranks. The copies are
    plain blocking transfers; no stream-ordering assumptions.

    Parameters the model reports via ``keep_in_fp32_for_weight_transfer`` skip the
    downcast. Those lists exist because the inference engine's kernels require fp32
    for them: vLLM's DeepSeek V4 hyper-connection kernels, for instance, assert
    ``fn``/``base``/``scale`` are fp32, and its loader silently casts a bf16 tensor
    back up rather than failing, so a downcast here shows up as lost mantissa in a
    Sinkhorn normalization rather than as an error.

    ``copy`` makes every returned tensor its own memory even when the parameter already
    lives on CPU, for a caller that reads the slice while training goes on.
    """
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    world = get_world()
    owners = partition_weights(model.state_dict(), world.world_size, dtype)
    partial: dict[str, Tensor] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                target_dtype = resolve_wire_dtype(keep_in_fp32, key, dtype)
                value = cast(DTensor, value.to(target_dtype)).full_tensor()
            if owners[key] != world.rank:
                continue
            partial[resolve_fqn(model, key)] = value.to("cpu", copy=copy)
        dist.barrier()

    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in partial.keys()):
        raise ValueError("gather_weights_parallel does not support LoRA state dicts")

    return partial


ShardList = list[tuple[str, dict[str, int]]]
"""One rank's temporary shards, in its write order: (file name, {tensor name: bytes})."""


def write_rank_shards(
    state_dict: dict[str, Tensor], save_dir: Path, rank: int, save_fn: Callable = save_file
) -> ShardList:
    """Write one rank's slice as <=5GB safetensors shards under temporary names."""
    tmp_pattern = f"tmp-rank{rank}{{suffix}}.safetensors"
    split = split_torch_state_dict_into_shards(state_dict, filename_pattern=tmp_pattern)
    local_shards: ShardList = []
    for tmp_name, tensor_names in split.filename_to_tensors.items():
        if not tensor_names:
            continue
        shard = {name: state_dict[name].contiguous() for name in tensor_names}
        save_fn(shard, save_dir / tmp_name, metadata={"format": "pt"})
        local_shards.append((tmp_name, {name: shard[name].nbytes for name in tensor_names}))
    return local_shards


def name_global_shards(all_shards: list[ShardList]) -> tuple[list[tuple[int, str, str]], dict[str, str], int]:
    """Number every rank's temporary shards into one checkpoint: rank order, then each rank's
    own order. Returns (rank, temporary name, final name) per shard, the index's weight map
    and the total size in bytes. Pure, so ranks that compute it apart agree."""
    flat = [(rank, tmp_name, sizes) for rank, shards in enumerate(all_shards) for tmp_name, sizes in shards]
    num_shards = len(flat)
    renames: list[tuple[int, str, str]] = []
    weight_map: dict[str, str] = {}
    for index, (rank, tmp_name, sizes) in enumerate(flat, start=1):
        if num_shards == 1:
            final_name = SAFE_WEIGHTS_NAME
        else:
            final_name = SAFE_WEIGHTS_NAME.replace(".safetensors", f"-{index:05d}-of-{num_shards:05d}.safetensors")
        for name in sizes:
            weight_map[name] = final_name
        renames.append((rank, tmp_name, final_name))
    total_size = sum(size for _, _, sizes in flat for size in sizes.values())
    return renames, weight_map, total_size


def write_weights_index(save_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    index_json = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(save_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
        f.write(json.dumps(index_json, indent=2, sort_keys=True) + "\n")


def save_state_dict_parallel(state_dict: dict[str, Tensor], save_dir: Path) -> None:
    """Cooperatively save a rank-partitioned state dict as sharded safetensors.

    Each rank splits its slice into <=5GB shards and writes them under temporary
    names, shard maps are exchanged, and every rank renames its own files into the
    global numbering (non-POSIX shared filesystems cannot reliably rename another
    rank's unpublished writes). Master writes the index last, after the barrier, so
    a complete index implies complete shards.
    """
    logger = get_logger()
    world = get_world()

    # Master-only mkdir + barrier: concurrent mkdir from every rank can re-raise
    # FileExistsError on a parallel FS (EEXIST + stale is_dir()).
    if world.is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    local_shards = write_rank_shards(state_dict, save_dir, world.rank)

    all_shards: list[ShardList | None] = [None] * world.world_size
    dist.all_gather_object(all_shards, local_shards)
    renames, weight_map, total_size = name_global_shards(cast(list[ShardList], all_shards))
    for rank, tmp_name, final_name in renames:
        if rank == world.rank:
            (save_dir / tmp_name).rename(save_dir / final_name)
    logger.debug(f"Saved {len(renames)} weight shards across {world.world_size} ranks")

    dist.barrier()
    if world.is_master and len(renames) > 1:
        write_weights_index(save_dir, weight_map, total_size)


# safetensors' Dtype enum, in its order: a file lists its tensors by descending dtype, then by name.
_SAFETENSORS_DTYPES: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.float8_e5m2: "F8_E5M2",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.int16: "I16",
    torch.uint16: "U16",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int32: "I32",
    torch.uint32: "U32",
    torch.float32: "F32",
    torch.complex64: "C64",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.uint64: "U64",
}
_SAFETENSORS_DTYPE_RANK = {dtype: rank for rank, dtype in enumerate(_SAFETENSORS_DTYPES)}


def _write_all(f, data, chunk_bytes: int) -> None:
    view = memoryview(data).cast("B")
    pos = 0
    while pos < len(view):
        pos += f.write(view[pos : pos + chunk_bytes])


def save_file_releasing_gil(
    tensors: dict[str, Tensor], path: Path, metadata: dict[str, str] | None = None, chunk_bytes: int = 1 << 28
) -> None:
    """``safetensors.torch.save_file`` for contiguous CPU tensors, byte for byte, written by
    plain file writes that release the GIL.

    safetensors 0.7 serializes and writes a file inside one call that holds the GIL (a 4 GiB
    shard stalled every other thread of the process for 15 of its 18 s), so a background writer
    built on it would stop the training loop. Anything outside the layout reproduced here
    (another dtype, a non-CPU or non-contiguous tensor, more than one metadata key, whose
    order safetensors leaves to a hash map) goes through ``save_file``.
    """
    if (
        sys.byteorder != "little"
        or (metadata is not None and len(metadata) > 1)
        or any(
            t.dtype not in _SAFETENSORS_DTYPE_RANK or t.device.type != "cpu" or not t.is_contiguous()
            for t in tensors.values()
        )
    ):
        save_file(tensors, path, metadata=metadata)
        return
    names = sorted(tensors, key=lambda name: (-_SAFETENSORS_DTYPE_RANK[tensors[name].dtype], name))
    header: dict = {} if metadata is None else {"__metadata__": metadata}
    offset = 0
    for name in names:
        tensor = tensors[name]
        nbytes = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    with open(path, "wb", buffering=0) as f:
        _write_all(f, struct.pack("<Q", len(header_bytes)) + header_bytes, chunk_bytes)
        for name in names:
            if tensors[name].numel():
                _write_all(f, tensors[name].reshape(-1).view(torch.uint8).numpy(), chunk_bytes)


HANDSHAKE_DIR = ".export_handshake"
EXPORT_HANDSHAKE_TIMEOUT_S = 3600.0


def _copy_base_files(base_dir: Path, save_dir: Path) -> None:
    for path in base_dir.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != SAFE_WEIGHTS_INDEX_NAME:
            shutil.copy2(path, save_dir / path.name)


def _publish_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


def _wait_for_files(paths: list[Path], handshake: Path, timeout: float, poll: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        errors = sorted(handshake.glob("error-rank*"))
        if errors:
            raise RuntimeError(f"weight export failed on another rank: {errors[0].read_text()}")
        if all(path.exists() for path in paths):
            return
        if time.monotonic() > deadline:
            missing = [path.name for path in paths if not path.exists()]
            raise TimeoutError(f"weight export waited {timeout:.0f}s for {missing} in {handshake}")
        time.sleep(poll)


def write_export_files(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    base_dir: Path,
    rank: int,
    world_size: int,
    timeout: float = EXPORT_HANDSHAKE_TIMEOUT_S,
    poll: float = 0.5,
) -> None:
    """The file half of an asynchronous export: what ``save_state_dict_parallel`` and the
    master's tail of ``export_policy_weights`` do, with a file handshake in
    ``save_dir/.export_handshake`` in place of torch.distributed, so it can run on a
    background thread. Each rank writes its temporary shards and publishes their list,
    waits for every rank's list, renames its own files into the global numbering
    (``name_global_shards``, as the synchronous path) and marks itself done; the master
    waits for every done marker, writes the index and the base model's files, removes the
    handshake and touches ``.complete`` last. A rank that fails leaves an error marker, so
    the others stop waiting. Empties ``state_dict`` once its shards are written."""
    handshake = save_dir / HANDSHAKE_DIR
    try:
        local_shards = write_rank_shards(state_dict, save_dir, rank, save_fn=save_file_releasing_gil)
        state_dict.clear()
        _publish_json(handshake / f"shards-rank{rank}.json", local_shards)
        lists = [handshake / f"shards-rank{r}.json" for r in range(world_size)]
        _wait_for_files(lists, handshake, timeout, poll)
        all_shards = [[(tmp, sizes) for tmp, sizes in json.loads(path.read_text())] for path in lists]
        renames, weight_map, total_size = name_global_shards(all_shards)
        for owner, tmp_name, final_name in renames:
            if owner == rank:
                (save_dir / tmp_name).rename(save_dir / final_name)
        (handshake / f"done-rank{rank}").touch()
        if rank == 0:
            _wait_for_files([handshake / f"done-rank{r}" for r in range(world_size)], handshake, timeout, poll)
            if len(renames) > 1:
                write_weights_index(save_dir, weight_map, total_size)
            _copy_base_files(base_dir, save_dir)
            shutil.rmtree(handshake)
            (save_dir / ".complete").touch()
    except BaseException as error:
        try:
            (handshake / f"error-rank{rank}").write_text(f"rank {rank}: {error!r}")
        except OSError:
            pass
        raise


class _PendingExport:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None


_pending_export: _PendingExport | None = None


def wait_for_weight_export() -> float:
    """Block until the files of the asynchronous export in flight, if any, are written, and
    re-raise its error. Returns the seconds waited."""
    global _pending_export
    pending, _pending_export = _pending_export, None
    if pending is None or pending.thread is None:
        return 0.0
    start = time.perf_counter()
    pending.thread.join()
    waited = time.perf_counter() - start
    if waited > 1:
        get_logger().info(f"Waited {waited:.1f}s for the weight export to {pending.save_dir} to finish writing")
    if pending.error is not None:
        raise RuntimeError(f"asynchronous weight export to {pending.save_dir} failed") from pending.error
    return waited


def _start_async_export(state_dict: dict[str, Tensor], save_dir: Path, base_dir: Path) -> None:
    global _pending_export
    world = get_world()
    if world.is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
        # A marker from an earlier export of this step (a resumed run) must not vouch for
        # the files written from now on.
        (save_dir / ".complete").unlink(missing_ok=True)
        shutil.rmtree(save_dir / HANDSHAKE_DIR, ignore_errors=True)
        (save_dir / HANDSHAKE_DIR).mkdir()
    dist.barrier()
    pending = _PendingExport(save_dir)

    def run() -> None:
        start = time.perf_counter()
        try:
            write_export_files(state_dict, save_dir, base_dir, world.rank, world.world_size)
        except BaseException as error:
            pending.error = error
            get_logger().error(f"Asynchronous weight export to {save_dir} failed: {error!r}")
        else:
            get_logger().info(f"Wrote policy weights to {save_dir} in {time.perf_counter() - start:.1f}s (background)")

    pending.thread = threading.Thread(target=run, name=f"weight-export-{save_dir.name}", daemon=True)
    pending.thread.start()
    _pending_export = pending


def export_policy_weights(model: nn.Module, base_model: str, save_dir: Path, async_save: bool = False) -> None:
    """Save the policy as a servable HF checkpoint in ``save_dir``: the weights as the
    filesystem broadcast writes them (every rank gathers and saves its slice), then the
    base model's non-weight files (config, generation config, tokenizer), then a
    ``.complete`` marker, so a reader that sees the marker sees a whole checkpoint.

    With ``async_save`` the gather and the HF conversion still run here, since they are
    collectives and must stay on the main thread in step order; the files are then written
    by a background thread per rank (``write_export_files``) while training goes on.
    ``wait_for_weight_export`` joins it; every export calls it first, so at most one
    export is in flight."""
    base_dir = Path(base_model)
    if not base_dir.is_dir():
        raise ValueError(f"weight export needs the base model as a local directory, got {base_model!r}")
    wait_for_weight_export()
    world = get_world()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    state_dict = gather_weights_parallel(model, copy=async_save)
    state_dict = convert_state_dict_to_hf(model, state_dict)
    if async_save:
        _start_async_export(state_dict, save_dir, base_dir)
        return
    save_state_dict_parallel(state_dict, save_dir)
    del state_dict
    if world.is_master:
        _copy_base_files(base_dir, save_dir)
        (save_dir / ".complete").touch()
    dist.barrier()
