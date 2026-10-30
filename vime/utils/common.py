import importlib

import torch

def is_npu() -> bool:
    if not hasattr(torch, "npu"):
        # Fresh Ray workers do not inherit the driver's torch_npu registration.
        try:
            importlib.import_module("torch_npu")
        except ModuleNotFoundError as exc:
            if exc.name != "torch_npu":
                raise
            return False

    if not torch.npu.is_available():
        raise RuntimeError(
            "torch_npu detected, but NPU device is not available or visible."
        )

    return True
