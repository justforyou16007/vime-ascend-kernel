"""Explicit hardware check: real Megatron load -> production raw HF iterator.

Uses the repository's model provider and mbridge checkpoint loader, without DDP
or optimizer allocation. Supply scripts/models/qwen3.5-35B-A3B.sh MODEL_ARGS.
"""

import faulthandler
import json
import os
import re
import runpy
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def load_training_model(*, wrap_with_ddp=False):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    # isort: split
    import mindspeed.megatron_adaptor  # noqa: F401
    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.training import get_model
    from mindspeed.args_utils import get_full_args
    from mindspeed.megatron_adaptor import repatch

    import vime_plugins.mbridge.qwen3_5  # noqa: F401
    from mbridge import AutoBridge
    from vime.backends.megatron_utils.arguments import set_default_megatron_args
    from vime.backends.megatron_utils.initialize import init
    from vime.backends.megatron_utils.model_provider import get_model_provider_func
    from vime.utils.distributed_utils import init_gloo_group

    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    add_args = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/convert_hf_to_torch_dist.py"))[
        "add_convertion_args"
    ]
    args = set_default_megatron_args(parse_args(add_args))
    validate_args(args)
    repatch(args)
    args = get_full_args()
    init(args)
    init_gloo_group()
    started = time.monotonic()
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=wrap_with_ddp)
    print(f"RANK {rank}: model built, allocated={torch.npu.memory_allocated()/2**30:.2f} GiB", flush=True)
    faulthandler.dump_traceback_later(90, repeat=True)
    print(f"RANK {rank}: constructing bridge", flush=True)
    bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    print(f"RANK {rank}: bridge ready, loading {sum(1 for m in model for _ in m.parameters())} parameters", flush=True)
    original_convert = bridge._weight_to_mcore_format
    loaded_count = 0

    def progress_convert(name, tensors):
        nonlocal loaded_count
        loaded_count += 1
        if loaded_count == 1 or loaded_count % 256 == 0:
            print(f"RANK {rank}: loading parameter {loaded_count}: {name}", flush=True)
        return original_convert(name, tensors)

    bridge._weight_to_mcore_format = progress_convert
    # mbridge copies into existing tensors; cache their views to avoid rebuilding 20K keys per parameter.
    bridge_model = model
    if wrap_with_ddp:
        from megatron.training.utils import unwrap_model

        bridge_model = unwrap_model(model)
    with ExitStack() as stack:
        for module in bridge_model:
            stack.enter_context(patch.object(module, "state_dict", return_value=module.state_dict()))
        bridge.load_weights(bridge_model, args.hf_checkpoint, memory_efficient=True)
    faulthandler.cancel_dump_traceback_later()
    print(f"RANK {rank}: model loaded in {time.monotonic()-started:.1f}s", flush=True)
    args.update_weight_buffer_size = 128 * 1024**2
    return args, model


def main():
    import torch
    import torch.distributed as dist
    from safetensors import safe_open

    args, model = load_training_model()
    if os.environ.get("VIME_VALIDATE_COLOCATE") == "1":
        from repro_megatron028_colocate import validate

        validate(args, model)
        dist.destroy_process_group()
        return

    from vime.backends.megatron_utils.update_weight.common import named_params_and_buffers
    from vime.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    rank = dist.get_rank()
    started = time.monotonic()
    iterator = HfWeightIteratorDirect(args, model, model_name="qwen3_5_moe", quantization_config=None)
    weights = dict(named_params_and_buffers(args, model))
    index = json.loads((Path(args.hf_checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {name for name in index if name.startswith("model.language_model.") or name == "lm_head.weight"}
    seen, differences, casts = set(), [], []
    covered = {name: set() for name in expected}
    with torch.no_grad():
        for chunk in iterator.get_hf_weight_chunks(weights):
            for name, tensor in chunk:
                assert name not in seen, f"Duplicate tensor {name}"
                seen.add(name)
                expert = re.fullmatch(r"(.+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
                if expert:
                    prefix, expert_id, projection = expert.groups()
                    key = prefix + (".down_proj" if projection == "down_proj" else ".gate_up_proj")
                    with safe_open(str(Path(args.hf_checkpoint) / index[key]), framework="pt") as f:
                        reference = f.get_slice(key)[int(expert_id)]
                    if projection != "down_proj":
                        reference = reference.chunk(2, dim=0)[projection == "up_proj"]
                    covered[key].add((int(expert_id), projection))
                else:
                    assert name in expected, f"Unexpected tensor {name}"
                    with safe_open(str(Path(args.hf_checkpoint) / index[name]), framework="pt") as f:
                        reference = f.get_tensor(name)
                    covered[name].add("full")
                actual = tensor.cpu()
                assert actual.shape == reference.shape, (name, actual.shape, reference.shape)
                if actual.dtype != reference.dtype:
                    casts.append({"name": name, "source": str(reference.dtype), "converted": str(actual.dtype)})
                if not torch.equal(actual, reference.to(actual.dtype)):
                    differences.append(
                        {"name": name, "max_abs": (actual.float() - reference.float()).abs().max().item()}
                    )
                del actual, reference
            del chunk
    missing = []
    for name, slices in covered.items():
        if "full" in slices:
            continue
        expected_slices = (
            (args.num_experts * (2 if name.endswith(".gate_up_proj") else 1)) if ".mlp.experts." in name else 1
        )
        if len(slices) != expected_slices:
            missing.append(name)
    report = {
        "rank": rank,
        "tp": args.tensor_model_parallel_size,
        "pp": args.pipeline_model_parallel_size,
        "tensors": len(seen),
        "missing": missing,
        "checkpoint_tensors": len(expected),
        "differences": differences,
        "dtype_changes": casts,
        "conversion_seconds": time.monotonic() - started,
        "peak_npu_allocated_gib": torch.npu.max_memory_allocated() / 2**30,
    }
    out = Path(os.environ["VIME_VALIDATION_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"conversion-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps({k: v for k, v in report.items() if k != "dtype_changes"}), flush=True)
    assert not missing, f"Missing {missing}"
    assert not differences, f"{len(differences)} tensors differ; see report"
    dist.barrier()
    dist.destroy_process_group()
    print(f"RANK {rank}: PASS full checkpoint round-trip through Megatron and production raw conversion", flush=True)


if __name__ == "__main__":
    main()
