"""Real two-process IPC/reload check; run explicitly on two reserved visible NPUs.

ASCEND_RT_VISIBLE_DEVICES=12,13 python tests/repro_npu_ipc_028.py
Only the second physical NPU allocates tensors; the receiver exposes just that NPU.
"""

import json
import multiprocessing as mp
import os
import traceback
from types import SimpleNamespace


def receive(pipe):
    if not pipe.poll(180):
        raise TimeoutError("IPC peer did not acknowledge within 180 seconds")
    result = pipe.recv()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


def receiver(pipe, devices):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = devices[-1]
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    try:
        import torch
        import torch_npu  # noqa: F401
        from vllm.config import VllmConfig
        from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

        from vime.backends.vllm_utils.npu_weight_transfer import VimeNPUIPCWeightTransferEngine, npu_generate_uuid

        index = 0
        torch.npu.set_device(index)
        device = torch.device("npu", index)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(16, 16, bias=True, device=device, dtype=torch.bfloat16)

            def load_weights(self, weights):
                for name, value in weights:
                    param = self.get_parameter(name)
                    param.weight_loader(param, value)

        model = Model()
        with torch.device(device):
            record_metadata_for_reloading(model)
        config = VllmConfig()
        config.model_config = SimpleNamespace(dtype=torch.bfloat16, multimodal_config=None)
        engine = VimeNPUIPCWeightTransferEngine(None, config, device, model)
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
        pipe.send({"ready": True, "uuid": npu_generate_uuid(index), "logical_device": index})
        for value in (2, 4):
            engine.start_weight_update()
            for _ in range(2):
                engine.update_weights(receive(pipe))
                pipe.send("ack")
            engine.finish_weight_update()
            assert torch.equal(model.layer.weight.cpu(), torch.full((16, 16), value, dtype=torch.bfloat16))
            assert torch.equal(model.layer.bias.cpu(), torch.full((16,), value + 1, dtype=torch.bfloat16))
            assert pointers == {n: p.data_ptr() for n, p in model.named_parameters()}
            pipe.send({"round": value // 2, "weights": "correct", "storage": "preserved"})
        engine.start_weight_update()
        engine.update_weights(receive(pipe))
        try:
            engine.finish_weight_update()
        except RuntimeError as error:
            assert "Incomplete" in str(error)
        else:
            raise AssertionError("Missing bias was accepted")
        try:
            engine.start_weight_update()
        except RuntimeError as error:
            assert "restart" in str(error)
        else:
            raise AssertionError("Failed engine accepted another update")
        pipe.send({"incomplete_update": "rejected", "retry": "requires_restart"})
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
        raise
    finally:
        pipe.close()


def main():
    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    assert len(devices) == 2 and len(set(devices)) == 2, "Supply two distinct reserved physical NPUs"
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=receiver, args=(child, devices))
    process.start()
    child.close()
    try:
        import base64

        import cloudpickle
        import torch
        import torch_npu  # noqa: F401
        from torch.multiprocessing.reductions import reduce_tensor
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        torch.npu.set_device(1)
        uuid = npu_generate_uuid(1)
        ready = receive(parent)
        assert ready["uuid"] == uuid and ready["logical_device"] == 0
        print(json.dumps(ready), flush=True)

        def send(name, tensor):
            torch.npu.synchronize()
            _, args = reduce_tensor(tensor)
            parent.send(
                {
                    "names": [name],
                    "dtype_names": ["bfloat16"],
                    "shapes": [list(tensor.shape)],
                    "ipc_handles_pickled": base64.b64encode(cloudpickle.dumps([{uuid: args}])).decode(),
                }
            )
            return receive(parent)

        for value in (2, 4):
            weight = torch.full((16, 16), value, dtype=torch.bfloat16, device="npu:1")
            bias = torch.full((16,), value + 1, dtype=torch.bfloat16, device="npu:1")
            assert send("layer.weight", weight) == "ack"
            weight.fill_(float("nan"))
            torch.npu.synchronize()
            assert send("layer.bias", bias) == "ack"
            print(json.dumps(receive(parent)), flush=True)
            del weight, bias
            torch.npu.ipc_collect()
        weight = torch.ones((16, 16), dtype=torch.bfloat16, device="npu:1")
        print(json.dumps(send("layer.weight", weight)), flush=True)
        process.join(30)
        assert process.exitcode == 0, process.exitcode
        print("PASS: real NPU IPC, logical remapping, two reloads, retained clones and failure isolation", flush=True)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


if __name__ == "__main__":
    main()
