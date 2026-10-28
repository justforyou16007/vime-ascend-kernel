"""
Colocated vLLM weight sync (trainer + worker)
=============================================

Trainer: ``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC (Ray).

Worker: ``vime.backends.vllm_utils.npu_weight_transfer`` owns checkpoint reloads.
The trainer retains every export until all ranks acknowledge the chunk.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    _begin_vllm_weight_update_session,
    _end_vllm_weight_update_session,
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


def _device_module() -> Any:
    """Return ``torch.npu`` if NPU is available, else ``torch.cuda``.

    This avoids importing ``mindspeed.megatron_adaptor`` in the vLLM
    subprocess (it breaks ``torch.compile``'s ``aot_compile``).
    """
    try:
        import torch_npu  # noqa: F401
        return torch.npu
    except ImportError:
        return torch.cuda


def _current_gpu_uuid() -> str:
    dev = _device_module()
    device_index = dev.current_device()
    if dev is getattr(torch, "npu", None):
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        return npu_generate_uuid(device_index)
    return str(dev.get_device_properties(device_index).uuid)


def _build_ipc_update_info_from_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, list], list[torch.Tensor]]:
    """Export rebuild arguments keyed by the producing rank's physical device ID.

    The receiver selects its physical device and remaps the logical device index.
    Return owning references: exports must live until every receiver acknowledges.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    ipc_handles: list[dict[str, tuple]] = []
    weight_refs: list[torch.Tensor] = []
    gpu_uuid = _current_gpu_uuid()

    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        weight = tensor.detach().contiguous()
        weight_refs.append(weight)
        _, ipc_args = reduce_tensor(weight)
        ipc_handles.append({gpu_uuid: ipc_args})

    return (
        {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        },
        weight_refs,
    )


def _serialize_ipc_update_info(info: dict[str, list]) -> str:
    """Pickle IPC handles for cross-rank gather (Gloo ``all_gather_object`` cannot carry them)."""
    import base64

    import cloudpickle

    return base64.b64encode(cloudpickle.dumps(info)).decode("ascii")


def _deserialize_ipc_update_info(payload: str) -> dict[str, list]:
    import base64

    import cloudpickle

    return cloudpickle.loads(base64.b64decode(payload.encode("ascii")))


def _merge_ipc_update_infos(infos: Sequence[dict[str, list]]) -> dict[str, list]:
    """Merge per-rank IPC payloads so each weight has handles for every GPU UUID in the slot."""
    if not infos:
        raise ValueError("no IPC update_info payloads to merge")
    base = infos[0]
    merged_handles: list[dict[str, tuple]] = []
    num_params = len(base["names"])
    for info in infos:
        if any(info[key] != base[key] for key in ("names", "dtype_names", "shapes")):
            raise ValueError("IPC ranks disagree on chunk names, dtypes or shapes")
        if len(info["ipc_handles"]) != num_params:
            raise ValueError("IPC handle count does not match parameter count")
    for i in range(num_params):
        combined: dict[str, tuple] = {}
        for info in infos:
            if combined.keys() & info["ipc_handles"][i].keys():
                raise ValueError("Multiple trainer ranks exported the same physical device")
            combined.update(info["ipc_handles"][i])
        merged_handles.append(combined)
    return {
        "names": base["names"],
        "dtype_names": base["dtype_names"],
        "shapes": base["shapes"],
        "ipc_handles": merged_handles,
    }


def count_colocated_engines(
    engine_gpu_offsets: Sequence[int], engine_gpu_counts: Sequence[int], total_actor_gpus: int
) -> int:
    """共卡(IPC)引擎数:GPU 槽位完全落在 actor 卡范围内的前缀引擎个数。

    槽位在范围内的引擎走 IPC(与 actor 同节点);第一个越界的引擎及其后全部走
    HCCL。专用段必须从越界处开始且连续(布局校验保证共卡段在前)。
    """
    colocate_engine_nums = 0
    for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
        if gpu_offset + gpu_count > total_actor_gpus:
            break
        colocate_engine_nums += 1
    return colocate_engine_nums


def _resolve_colocated_engine_count(
    args, engine_gpu_offsets: Sequence[int], engine_gpu_counts: Sequence[int]
) -> int:
    """Resolve the IPC-engine prefix from physical engine roles."""
    from vime.ray.engine_roles import (
        EngineRole,
        EngineRoleError,
        colocated_prefix_count,
        resolve_engine_roles,
    )

    total_actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    spec = getattr(args, "resource_layout_spec", None)
    hybrid = spec is not None and getattr(spec, "rollout_has_share", False)
    try:
        colocation = {role.gpu_slot: role.colocated for role in resolve_engine_roles(args)}
    except EngineRoleError:
        if hybrid:
            raise
        logger.warning("engine role resolution failed; falling back to the legacy slot-range heuristic")
        return count_colocated_engines(engine_gpu_offsets, engine_gpu_counts, total_actor_gpus)

    verdicts = [colocation.get(offset) for offset in engine_gpu_offsets]
    if any(verdict is None for verdict in verdicts):
        missing = [offset for offset, verdict in zip(engine_gpu_offsets, verdicts, strict=True) if verdict is None]
        if hybrid:
            raise EngineRoleError(
                f"engine roles do not cover hybrid rollout GPU slots {missing}; "
                "cannot safely split IPC and distributed weight sync"
            )
        logger.info(
            "engine roles do not cover GPU slots %s (heterogeneous per-group engine sizes); "
            "falling back to the legacy slot-range heuristic",
            missing,
        )
        return count_colocated_engines(engine_gpu_offsets, engine_gpu_counts, total_actor_gpus)

    roles = tuple(
        EngineRole(index=i, gpu_slot=offset, placement=(), colocated=bool(verdict))
        for i, (offset, verdict) in enumerate(zip(engine_gpu_offsets, verdicts, strict=True))
    )
    return colocated_prefix_count(roles)


def _actor_ranks_by_gpu_slot(args) -> dict[int, tuple[int, ...]]:
    """Return actor ranks for each colocated rollout engine GPU slot."""
    from vime.ray.engine_roles import EngineRoleError, resolve_engine_roles

    spec = getattr(args, "resource_layout_spec", None)
    hybrid = spec is not None and getattr(spec, "rollout_has_share", False)
    try:
        roles = resolve_engine_roles(args)
    except EngineRoleError:
        if hybrid:
            raise
        logger.warning("engine role resolution failed; falling back to slot-as-rank IPC groups")
        return {}
    return {role.gpu_slot: role.actor_ranks for role in roles if role.colocated}


def _run_weight_update_step(action):
    """Keep exports alive and report RPC failures to every trainer rank."""
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    group = get_gloo_group()
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failures = [item for item in errors if item is not None]
    if failures:
        raise RuntimeError("Weight update failed; rollout stays paused: " + "; ".join(failures))
    return result


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → all_gather_object(Gloo CPU, over the engine
    slot ranks) → Ray IPC to engine.  Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        # shortcut: raw provides identical buckets on every rank; validate bridge
        # collective ordering before enabling it with per-chunk acknowledgements.
        if args.megatron_to_hf_mode != "raw":
            raise ValueError("Colocated vLLM 0.28 weight sync requires --megatron-to-hf-mode raw")

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        # 混合分发(共卡 IPC + 远程 HCCL)的分布式半边状态;connect_rollout_engines 填充。
        self.distributed_rollout_engines: list = []
        self.use_distribute = False
        self._is_distributed_src_rank = False
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # vLLM IPC handle payloads may use cloudpickle on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        colocate_engine_nums = _resolve_colocated_engine_count(
            self.args, engine_gpu_offsets, engine_gpu_counts
        )

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "vime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        ranks_by_slot = _actor_ranks_by_gpu_slot(self.args)

        def _actor_ranks_for(i: int) -> list[int]:
            ranks = ranks_by_slot.get(colocate_gpu_offsets[i])
            if ranks:
                return list(ranks)
            return list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = _actor_ranks_for(i)
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = min(group_ranks)

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            if dist.get_rank() in _actor_ranks_for(i):
                self._ipc_engine = engine

        # vLLM #39212: one-time IPC transfer-engine init on each colocated engine.
        if dist.get_rank() == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self.rollout_engines])
            self._ipc_initialized = True

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        next_version = self.weight_version + 1

        # 混合分发:pause/flush/continue 与会话管理必须同时覆盖共卡(IPC)与远程(HCCL)
        # 引擎 —— connect_rollout_engines 之后 self.rollout_engines 只剩共卡子集。
        all_engines = list(self.rollout_engines) + list(self.distributed_rollout_engines)

        rank = dist.get_rank()
        _run_weight_update_step(
            lambda: ray.get([engine.pause_generation.remote() for engine in all_engines]) if rank == 0 else None
        )
        _run_weight_update_step(
            lambda: ray.get([engine.flush_cache.remote() for engine in all_engines]) if rank == 0 else None
        )
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=all_engines,
                )
        dist.barrier(group=get_gloo_group())

        # vLLM #39212: enter weight-update mode on each slot leader.
        _run_weight_update_step(
            lambda: ray.get(self._ipc_engine.start_weight_update.remote())
            if self._ipc_engine is not None and rank == self._ipc_gather_src else None
        )
        if self.use_distribute and self.distributed_rollout_engines:
            _begin_vllm_weight_update_session(self.distributed_rollout_engines)
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            def send_chunk(chunk=hf_named_tensors):
                refs, exports = self._send_hf_params(chunk, weight_version=next_version)
                ray.get(refs)
                return exports

            long_lived_tensors = _run_weight_update_step(send_chunk)
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (vLLM engines)
            # have already closed their IPC handles.
            del long_lived_tensors, hf_named_tensors, send_chunk
            _device_module().ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        _device_module().ipc_collect()

        # vLLM #39212: exit weight-update mode.
        if self.use_distribute and self.distributed_rollout_engines:
            _end_vllm_weight_update_session(self.distributed_rollout_engines)
        _run_weight_update_step(
            lambda: ray.get(self._ipc_engine.finish_weight_update.remote(weight_version=str(next_version)))
            if self._ipc_engine is not None and rank == self._ipc_gather_src else None
        )
        self.weight_version = next_version
        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=all_engines,
                )
        _run_weight_update_step(
            lambda: ray.get([engine.continue_generation.remote() for engine in all_engines]) if rank == 0 else None
        )
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors, *, weight_version) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
                packed=False,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # all_gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
        ref = ipc_engine.update_weights_from_tensor.remote(**local_info, weight_version=str(weight_version))
        return [ref], weight_refs

    local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
    payload = _serialize_ipc_update_info(local_info)

    # all_gather_object is monkey-patched for ReloadableProcessGroup; gather_object
    # is not (it fails after a Megatron reload).
    gathered_payloads = [None] * slot_size
    dist.all_gather_object(gathered_payloads, payload, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        merged = _merge_ipc_update_infos(slot_infos)
        refs.append(ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version)))

    return refs, weight_refs
