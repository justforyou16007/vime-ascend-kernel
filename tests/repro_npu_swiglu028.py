"""NPU SwiGLU check with changing expert token counts; --fused reproduces the old path."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(1234)
    if args.fused:
        from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

    rows = []
    for count in (821, 1033, 17, 641):
        x = torch.randn(count, 1024, device="npu", dtype=torch.bfloat16, requires_grad=True)
        weights = torch.rand(count, 1, device="npu", requires_grad=True)
        if args.fused:
            actual = weighted_bias_swiglu_impl(x, None, weights)
        else:
            # The existing TEGroupedMLP path selected by --no-bias-swiglu-fusion.
            gate, value = x.chunk(2, dim=-1)
            actual = (F.silu(gate) * value * weights).to(x.dtype)
        torch.npu.synchronize()
        rx = x.detach().float().requires_grad_()
        rw = weights.detach().clone().requires_grad_()
        gate, value = rx.chunk(2, dim=-1)
        reference = F.silu(gate) * value * rw
        torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.03)
        actual.float().sum().backward()
        reference.sum().backward()
        torch.npu.synchronize()
        torch.testing.assert_close(x.grad.float(), rx.grad, rtol=0.04, atol=0.04)
        torch.testing.assert_close(weights.grad, rw.grad, rtol=0.04, atol=0.1)
        rows.append({"tokens": count, "forward_backward": "passed"})
        args.output.write_text(json.dumps({"fused": args.fused, "results": rows}, indent=2))
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
