"""Small real-NPU CPU-Adam/state-handoff check, without DDP or a language model.

Run explicitly with ASCEND_RT_VISIBLE_DEVICES set to one reserved NPU.
This is a dependency/lifecycle check, not a substitute for the 35B training run.
"""

from types import SimpleNamespace


def main():
    import torch
    import torch_npu  # noqa: F401

    # isort: split
    import mindspeed.megatron_adaptor  # noqa: F401
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    from vime.utils.npu_training_state_offloader import NPUTrainingStateOffloader

    torch.npu.set_device(0)
    params = [torch.nn.Parameter(torch.ones(1024, device="npu:0", dtype=torch.bfloat16)) for _ in range(2)]
    optimizers = [
        HybridDeviceOptimizer(
            [param],
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=True,
            pin_cpu_grads=True,
            pin_cpu_params=True,
            lr=0.01,
            weight_decay=0.1,
            betas=(0.9, 0.98),
            eps=1e-8,
            fused=True,
        )
        for param in params
    ]
    wrapper = SimpleNamespace(optimizer=optimizers[0])
    offloader = NPUTrainingStateOffloader()
    for step in (1, 2):
        for param, optimizer in zip(params, optimizers, strict=True):
            optimizer.zero_grad()
            loss = param.float().square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
            before = param.detach().clone()
            optimizer.step()
            torch.npu.synchronize()
            assert not torch.equal(before, param), "Optimizer did not update weights"
        assert torch.equal(params[0], params[1]), "Handoff changed the next optimizer update"
        if step == 1:
            state_tensors = [
                value
                for optimizer in optimizers[0].cpu_optimizers
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ]
            assert state_tensors and all(t.device.type == "cpu" for t in state_tensors)
            snapshots = [(t, t.data_ptr(), t.clone()) for t in state_tensors]
            assert offloader.offload(wrapper) == 0, "CPU optimizer state was unnecessarily copied"
            offloader.onload(wrapper)
            for tensor, pointer, value in snapshots:
                assert tensor.device.type == "cpu" and tensor.data_ptr() == pointer
                assert torch.equal(tensor, value), "Optimizer state changed during handoff"
        print(f"PASS step {step}: finite gradients, CPU Adam update, matching uninterrupted control", flush=True)
    print("PASS: CPU optimizer state survives phase handoff; DDP and 35B remain untested", flush=True)


if __name__ == "__main__":
    main()
