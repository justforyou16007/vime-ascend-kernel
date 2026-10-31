"""Two identity checkpoint reloads through a running 0.28 HTTP server and real IPC.

Run explicitly with ASCEND_RT_VISIBLE_DEVICES matching the server's reserved cards.
The tested server command (after sourcing the migration environment) is:

ASCEND_RT_VISIBLE_DEVICES=12,13 VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:False \
vllm serve /home/docker/Qwen3.6-35B-A3B --served-model-name vime-028-validation \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 2048 --max-num-seqs 2 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.72 --enforce-eager \
  --enable-sleep-mode --language-model-only --distributed-executor-backend mp \
  --weight-transfer-config '{"backend":"vime_npu_ipc"}' \
  --worker-extension-cls vime.backends.vllm_utils.npu_weight_transfer.VimeNPUWorkerExtension \
  --additional-config '{"rl_config":{"enabled":true},"weight_nz_mode":0,"multistream_overlap_shared_expert":false}' \
  --host 127.0.0.1 --port 28728

Stop the test server after validation to release the reserved cards.
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=28728)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import requests
    import torch
    import torch_npu  # noqa: F401
    from safetensors import safe_open
    from torch.multiprocessing.reductions import reduce_tensor
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

    from vime.backends.vllm_utils.vllm_engine import VLLMEngine

    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    assert len(devices) == 2, "This validation profile requires TP=2"
    engine = VLLMEngine(SimpleNamespace(hf_checkpoint=args.model), rank=0)
    engine.server_host, engine.server_port = "127.0.0.1", args.port
    base = f"http://127.0.0.1:{args.port}"
    report = {"model": args.model, "visible_devices": devices, "rounds": []}

    def generate():
        response = requests.post(
            base + "/v1/completions",
            json={
                "model": "vime-028-validation",
                "prompt": "The capital of France is",
                "max_tokens": 16,
                "temperature": 0,
                "seed": 123,
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        assert result["usage"]["completion_tokens"] > 0
        text = result["choices"][0]["text"]
        assert text.strip(), result
        return text

    baseline = generate()
    report["baseline"] = baseline
    Path(args.output).write_text(json.dumps(report, indent=2))
    print("BASELINE", repr(baseline), flush=True)
    index = json.loads((Path(args.model) / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(n for n in index if n.startswith("model.language_model.") or n == "lm_head.weight")
    assert len(names) > 600, "Unexpected checkpoint layout"
    engine.init_weight_transfer_engine({"init_info": {"packed": False}})
    for version in (1, 2):
        started = time.monotonic()
        engine.pause_generation()
        engine.release_memory_occupation(level=2)
        print(f"ROUND {version}: level-2 sleep complete", flush=True)
        engine.resume_memory_occupation(tags=["weights"])
        engine.start_weight_update()
        for i, name in enumerate(names):
            with safe_open(str(Path(args.model) / index[name]), framework="pt") as shard:
                cpu = shard.get_tensor(name)
            exports, handles = [], {}
            for logical in range(len(devices)):
                torch.npu.set_device(logical)
                tensor = cpu.to(f"npu:{logical}").contiguous()
                exports.append(tensor)
                torch.npu.synchronize()
                _, ipc_args = reduce_tensor(tensor)
                handles[npu_generate_uuid(logical)] = ipc_args
            engine.update_weights_from_tensor(
                names=[name],
                dtype_names=[str(cpu.dtype).split(".")[-1]],
                shapes=[list(cpu.shape)],
                ipc_handles=[handles],
                weight_version=str(version),
            )
            del exports, tensor, cpu
            for logical in range(len(devices)):
                torch.npu.set_device(logical)
                torch.npu.ipc_collect()
            if i % 40 == 0 or i == len(names) - 1:
                print(f"ROUND {version}: {i + 1}/{len(names)} {name}", flush=True)
        engine.finish_weight_update(weight_version=str(version))
        engine.continue_generation()
        assert engine._resume_pending, "Generation resumed before KV wake"
        engine.resume_memory_occupation(tags=["kv_cache"])
        output = generate()
        assert output == baseline, {"baseline": baseline, "after_reload": output}
        result = {
            "version": version,
            "tensors": len(names),
            "seconds": time.monotonic() - started,
            "output": output,
            "matches_baseline": True,
        }
        report["rounds"].append(result)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print("ROUND PASS", json.dumps(result), flush=True)
    print("PASS: two level-2 sleep/full IPC reload/KV wake/generate cycles", flush=True)


if __name__ == "__main__":
    main()
