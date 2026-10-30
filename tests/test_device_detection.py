from types import SimpleNamespace

import pytest

from vime.utils import common


def test_fresh_worker_registers_npu(monkeypatch):
    torch = SimpleNamespace()
    monkeypatch.setattr(common, "torch", torch)

    def register(name):
        assert name == "torch_npu"
        torch.npu = SimpleNamespace(is_available=lambda: True)

    monkeypatch.setattr(common.importlib, "import_module", register)
    assert common.is_npu()


@pytest.mark.parametrize("missing", ["torch_npu", "broken_npu_dependency"])
def test_only_absent_extension_means_non_npu(monkeypatch, missing):
    monkeypatch.setattr(common, "torch", SimpleNamespace())

    def unavailable(name):
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(common.importlib, "import_module", unavailable)
    if missing == "torch_npu":
        assert not common.is_npu()
    else:
        with pytest.raises(ModuleNotFoundError):
            common.is_npu()


def test_unavailable_registered_npu_fails(monkeypatch):
    monkeypatch.setattr(common, "torch", SimpleNamespace(npu=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="not available"):
        common.is_npu()
