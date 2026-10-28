"""Checkpoint-format NPU IPC for vLLM 0.28, without importing the training stack."""

import torch
from torch_npu.multiprocessing.reductions import rebuild_npu_tensor
from vllm.config import set_current_vllm_config
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCWeightTransferEngine, npu_generate_uuid


class VimeNPUIPCWeightTransferEngine(NPUIPCWeightTransferEngine):
    """Own the reload lifecycle while NPUWorker owns the HTTP session state."""

    def __init__(self, config, vllm_config, device, model):
        super().__init__(config, vllm_config, device, model)
        # shortcut: full, unquantized text-model updates only; validate padding,
        # multimodal and draft-model loading before extending this backend.
        if vllm_config.quant_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("vime_npu_ipc requires unquantized weights and no speculative decoding")
        mm = self.model_config.multimodal_config
        if mm is not None and not mm.language_model_only:
            raise ValueError("vime_npu_ipc requires --language-model-only for multimodal checkpoints")
        self._failed = False
        self._seen_names = set()

    def init_transfer_engine(self, init_info):
        if init_info.packed:
            raise ValueError("vime_npu_ipc does not support packed transfers")
        super().init_transfer_engine(init_info)

    def _check_healthy(self):
        if self._failed:
            raise RuntimeError("Weight update failed; restart the rollout engine before retrying")

    def start_weight_update(self):
        self._check_healthy()
        try:
            for name, layer in self.model.named_modules():
                if any(True for _ in layer.parameters(recurse=False)) and layer not in LAYERWISE_INFO:
                    raise RuntimeError(f"Missing pre-load reload metadata for {name}")
            self._seen_names.clear()
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                initialize_layerwise_reload(self.model)
        except Exception:
            self._failed = True
            raise

    def update_weights(self, update_info):
        self._check_healthy()
        try:
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                super().update_weights(update_info)
        except Exception:
            self._failed = True
            raise

    def receive_weights(self, update_info):
        physical_id = npu_generate_uuid(self.device.index)
        if not isinstance(update_info.ipc_handles, list):
            raise ValueError("Expected unpacked per-tensor IPC handles")
        weights = []
        for name, shape, dtype, handles in zip(
            update_info.names, update_info.shapes, update_info.dtype_names, update_info.ipc_handles, strict=True
        ):
            if name in self._seen_names:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            if physical_id not in handles:
                raise ValueError(f"IPC handle missing for physical NPU {physical_id}")
            args = list(handles[physical_id])
            args[6] = self.device.index
            borrowed = rebuild_npu_tensor(*args)
            if list(borrowed.shape) != shape or str(borrowed.dtype).split(".")[-1] != dtype:
                raise ValueError(f"IPC tensor metadata mismatch: {name}")
            # Layerwise loaders retain unfinished layers beyond this RPC. Own the data
            # before the trainer releases/reuses its export after the acknowledgement.
            weights.append((name, borrowed.clone()))
            self._seen_names.add(name)
        self.model.load_weights(weights)

    def finish_weight_update(self):
        self._check_healthy()
        try:
            incomplete = []
            for name, layer in self.model.named_modules():
                # Native level-2 wake restores buffers (RoPE caches, unquantized
                # attention scales); they are not trainer checkpoint parameters.
                if not any(True for _ in layer.parameters(recurse=False)):
                    continue
                info = LAYERWISE_INFO.get(layer)
                if info is not None and info.can_load() and info.load_numel < info.load_numel_total:
                    incomplete.append(f"{name}: {info.load_numel}/{info.load_numel_total}")
            if incomplete:
                # Native finalize may restore old kernel storage for missing layers;
                # after level-2 sleep that storage has no valid weights to restore.
                raise RuntimeError("Incomplete checkpoint update: " + "; ".join(incomplete[:10]))
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                finalize_layerwise_reload(self.model, self.model_config)
                torch.accelerator.synchronize()
        except Exception:
            self._failed = True
            raise


class VimeNPUWorkerExtension:
    """Register before NPUWorker.load_model creates the transfer engine."""

    def __new__(cls, **kwargs):
        if "vime_npu_ipc" not in WeightTransferEngineFactory._registry:
            WeightTransferEngineFactory.register_engine("vime_npu_ipc", VimeNPUIPCWeightTransferEngine)
        return super().__new__(cls)
