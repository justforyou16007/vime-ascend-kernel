"""Exercise the installed vLLM reload code with CPU tensors and mocked NPU IPC."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

from vime.backends.vllm_utils import npu_weight_transfer as mod


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2, bias=True)

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            param = self.get_parameter(name)
            param.weight_loader(param, tensor)
            loaded.add(name)
        return loaded


@pytest.fixture
def engine(monkeypatch):
    model = ToyModel()
    record_metadata_for_reloading(model)
    config = SimpleNamespace(
        parallel_config=None,
        model_config=SimpleNamespace(dtype=torch.float32, multimodal_config=None),
        quant_config=None,
        speculative_config=None,
    )
    monkeypatch.setattr(mod, "set_current_vllm_config", lambda *_: __import__("contextlib").nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(mod, "npu_generate_uuid", lambda *_: "host-3")
    monkeypatch.setattr(mod, "rebuild_npu_tensor", lambda *args: args[0])
    mod.VimeNPUWorkerExtension()
    return WeightTransferEngineFactory.create_engine(
        WeightTransferConfig(backend="vime_npu_ipc"),
        config,
        torch.device("cpu"),
        model,
    )


def payload(name, tensor, uuid="host-3"):
    return dict(
        names=[name],
        dtype_names=[str(tensor.dtype).split(".")[-1]],
        shapes=[list(tensor.shape)],
        ipc_handles=[{uuid: (tensor, 0, 0, 0, 0, 0, 7)}],
    )


def test_two_rounds_keep_borrowed_weights_and_storage(engine):
    addresses = {n: p.data_ptr() for n, p in engine.model.named_parameters()}
    for value in (2.0, 4.0):
        weight, bias = torch.full((2, 2), value), torch.full((2,), value + 1)
        engine.start_weight_update()
        engine.update_weights(payload("layer.weight", weight))
        weight.fill_(float("nan"))  # The layer is incomplete until the next chunk.
        engine.update_weights(payload("layer.bias", bias))
        engine.finish_weight_update()
        assert torch.equal(engine.model.layer.weight, torch.full((2, 2), value))
        assert torch.equal(engine.model.layer.bias, bias)
        assert addresses == {n: p.data_ptr() for n, p in engine.model.named_parameters()}


def test_missing_weights_fail_closed(engine):
    engine.start_weight_update()
    engine.update_weights(payload("layer.weight", torch.ones(2, 2)))
    with pytest.raises(RuntimeError, match="Incomplete"):
        engine.finish_weight_update()
    with pytest.raises(RuntimeError, match="restart"):
        engine.start_weight_update()


def test_buffer_only_layers_keep_native_wake_restored_values(engine):
    engine.model.register_buffer("rotary_cache", torch.arange(4.0), persistent=False)
    record_metadata_for_reloading(engine.model)
    engine.start_weight_update()
    engine.update_weights(payload("layer.weight", torch.ones(2, 2)))
    engine.update_weights(payload("layer.bias", torch.ones(2)))
    engine.finish_weight_update()
    assert torch.equal(engine.model.rotary_cache, torch.arange(4.0))


@pytest.mark.parametrize("corruption", ["uuid", "shape", "dtype", "duplicate"])
def test_invalid_chunk_poison_session(engine, corruption):
    engine.start_weight_update()
    data = payload("layer.weight", torch.ones(2, 2))
    if corruption == "uuid":
        data["ipc_handles"][0] = {"other-host-3": next(iter(data["ipc_handles"][0].values()))}
    elif corruption == "shape":
        data["shapes"] = [[4]]
    elif corruption == "dtype":
        data["dtype_names"] = ["bfloat16"]
    else:
        engine.update_weights(data)
    with pytest.raises(ValueError):
        engine.update_weights(data)
    with pytest.raises(RuntimeError, match="restart"):
        engine.start_weight_update()


def test_packed_init_rejected(engine):
    with pytest.raises(ValueError, match="packed"):
        engine.init_transfer_engine(engine.parse_init_info({"packed": True}))


def test_serialized_payload_remaps_receiver_device(engine, monkeypatch):
    import base64
    import cloudpickle

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    seen = []
    monkeypatch.setattr(mod, "rebuild_npu_tensor", lambda *args: seen.append(args[6]) or args[0])
    engine.start_weight_update()
    for name, tensor in [("layer.weight", torch.ones(2, 2)), ("layer.bias", torch.ones(2))]:
        data = payload(name, tensor)
        data["ipc_handles_pickled"] = base64.b64encode(cloudpickle.dumps(data.pop("ipc_handles"))).decode()
        engine.update_weights(data)
    engine.finish_weight_update()
    assert seen == [engine.device.index, engine.device.index]


def test_missing_initial_metadata_rejected(engine):
    engine.model.extra = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="Missing pre-load reload metadata"):
        engine.start_weight_update()


def test_registration_is_idempotent_and_has_no_worker_method_conflicts(engine):
    from vllm_ascend.worker.worker import NPUWorker

    mod.VimeNPUWorkerExtension()
    attrs = {n for n in dir(mod.VimeNPUWorkerExtension) if not n.startswith("__")}
    assert not (attrs & set(dir(NPUWorker)))


def test_native_worker_lifecycle(engine, monkeypatch):
    from vllm_ascend.worker.worker import NPUWorker

    worker = object.__new__(NPUWorker)
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    monkeypatch.setattr(worker, "_check_nz_disabled", lambda: None)
    worker.start_weight_update()
    with pytest.raises(RuntimeError, match="already active"):
        worker.start_weight_update()
    worker.update_weights(payload("layer.weight", torch.ones(2, 2)))
    worker.update_weights(payload("layer.bias", torch.ones(2)))
    worker.finish_weight_update()
    assert not worker._weight_update_active
    with pytest.raises(RuntimeError, match="start_weight_update"):
        worker.update_weights(payload("layer.bias", torch.ones(2)))
