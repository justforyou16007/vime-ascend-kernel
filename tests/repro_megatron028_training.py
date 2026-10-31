"""Explicit 35B PP2/EP4 hardware check: two LM steps with a real state handoff.

Uses synthetic token sequences and LM cross entropy, not the full RL/Polar loop.
Run with run_megatron028_training.sh on eight reserved NPUs.
Set VIME_VALIDATE_COLOCATE=1 to run four TP2 rollout engines between steps.
"""

import dataclasses
import faulthandler
import json
import math
import os
import time
from pathlib import Path

from repro_megatron028_conversion import load_training_model


def main():
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from vime.utils.reloadable_process_group import (
        destroy_process_groups,
        monkey_patch_torch_dist,
        reload_process_groups,
    )

    monkey_patch_torch_dist()
    args, model = load_training_model(wrap_with_ddp=True)

    from megatron.core.distributed import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.utils import get_model_config

    from vime.backends.megatron_utils.update_weight.common import named_params_and_buffers
    from vime.utils.distributed_utils import get_gloo_group
    from vime.utils.memory_utils import clear_memory
    from vime.utils.npu_training_state_offloader import NPUTrainingStateOffloader
    from vime.utils.npu_weight_offloader import NPUWeightOffloader
    from vime.utils.tensor_backper import TensorBackuper

    assert dist.get_world_size() == 8
    assert (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.context_parallel_size,
        args.expert_model_parallel_size,
    ) == (1, 2, 1, 4)
    rank = dist.get_rank()
    out = Path(os.environ["VIME_VALIDATION_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    faulthandler.dump_traceback_later(180, repeat=True)
    config = OptimizerConfig(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(OptimizerConfig) if hasattr(args, f.name)}
    )
    config.timers = None
    optimizer = get_megatron_optimizer(config, model, use_gloo_process_groups=args.enable_gloo_process_groups)
    model_config = get_model_config(model[0])
    model_config.grad_scale_func = optimizer.scale_loss
    model_config.finalize_model_grads_func = finalize_model_grads
    model_config.timers = None
    assert not args.overlap_grad_reduce and not args.overlap_param_gather
    for module in model:
        module.train()
    print(f"RANK {rank}: optimizer ready, allocated={torch.npu.memory_allocated() / 2**30:.3f} GiB", flush=True)

    def forward_step(_iterator, module):
        # Identical inputs on PP peers; a small deterministic sequence exercises all layers.
        tokens = (torch.arange(args.seq_length, device="npu").unsqueeze(0) % 1000) + 100
        positions = torch.arange(args.seq_length, device="npu").unsqueeze(0)
        cu = torch.tensor([0, args.seq_length], device="npu", dtype=torch.int32)
        packed = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=args.seq_length,
            max_seqlen_kv=args.seq_length,
            qkv_format="thd",
        )
        output = module(
            input_ids=tokens, position_ids=positions, attention_mask=None, labels=tokens + 1, packed_seq_params=packed
        )

        def loss_func(losses):
            loss = losses.float().mean()
            assert torch.isfinite(loss), "Non-finite LM loss"
            return loss, {"lm_loss": loss.detach()}

        return output, loss_func

    backup = TensorBackuper.create(lambda: named_params_and_buffers(args, model), single_tag=None)
    weight_offloader = NPUWeightOffloader(release_param_buffer=True, param_restorer=lambda: backup.restore("actor"))
    state_offloader = NPUTrainingStateOffloader()
    # Check a full dense matrix on each stage; a small BF16 slice may round away a real update.
    probe_name, probe = next((n, p) for n, p in model[0].named_parameters() if "linear_attn.in_proj.weight" in n)
    report = {"rank": rank, "tp": 1, "pp": 2, "cp": 1, "ep": 4, "steps": [], "probe": probe_name}
    colocate = os.environ.get("VIME_VALIDATE_COLOCATE") == "1"

    def train_step(step):
        if step > 1:
            clear_memory()
            reload_process_groups()
            weight_offloader.onload(model)
            state_offloader.onload(optimizer)
            for name, param in named_params_and_buffers(args, model):
                assert torch.equal(param.detach().cpu(), backup.get("actor")[name]), name
            print(f"RANK {rank}: PASS exact parameter restoration after real DDP/state handoff", flush=True)
        started = time.monotonic()
        for module in model:
            module.zero_grad_buffer()
        optimizer.zero_grad()
        before = probe.detach().cpu().clone()
        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=None,
            model=model,
            num_microbatches=args.global_batch_size // args.data_parallel_size,
            seq_length=args.seq_length,
            micro_batch_size=1,
            forward_only=False,
        )
        print(f"RANK {rank}: step {step} backward complete", flush=True)
        success, grad_norm, _ = optimizer.step()
        torch.npu.synchronize()
        assert success and math.isfinite(float(grad_norm)) and float(grad_norm) > 0, (success, grad_norm)
        changed = int((before != probe.detach().cpu()).sum())
        assert changed > 0, f"No parameter update on rank {rank}"
        result = {
            "step": step,
            "grad_norm": float(grad_norm),
            "changed_probe_elements": changed,
            "losses": [{k: float(v) for k, v in loss.items()} for loss in losses],
            "seconds": time.monotonic() - started,
            "peak_npu_gib": torch.npu.max_memory_allocated() / 2**30,
        }
        report["steps"].append(result)
        print(f"RANK {rank}: STEP_RESULT {json.dumps(result)}", flush=True)
        del before, losses
        for module in model:
            module.zero_grad_buffer()
        optimizer.zero_grad()
        if step == 1 or colocate:
            backup.backup("actor")
            state_offloader.offload(optimizer, model=model)
            destroy_process_groups()
            report["released_flat_buffer_gib"] = weight_offloader.offload(model) / 2**30
            report["sleep_allocated_gib"] = torch.npu.memory_allocated() / 2**30
            assert report["released_flat_buffer_gib"] > 20, report
            assert report["sleep_allocated_gib"] < 2, report
            dist.barrier(group=get_gloo_group())
            if colocate:
                # Weight conversion needs PP/EP collectives while DDP buffers remain offloaded.
                reload_process_groups()
        (out / f"training-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    if colocate:
        from repro_megatron028_colocate import validate

        validate(args, model, backup=backup, train_step=train_step)
    else:
        for step in (1, 2):
            train_step(step)
    dist.barrier(group=get_gloo_group())
    faulthandler.cancel_dump_traceback_later()
    print(f"RANK {rank}: PASS two 35B LM training steps across state handoff", flush=True)
    destroy_process_groups()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
