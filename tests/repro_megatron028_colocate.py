"""Run via VIME_VALIDATE_COLOCATE=1 bash tests/run_megatron028_conversion.sh.

Exercises the real updater, Ray actors, HTTP adapter and TP2 IPC backends.
The conversion-only caller discards its model after a CPU backup. The training
caller supplies a step callback that performs real DDP/optimizer state handoff.
"""

import gc
import faulthandler
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def validate(args, model, *, backup=None, train_step=None):
    import ray
    import requests
    import torch
    import torch.distributed as dist

    from vime.backends.megatron_utils.update_weight.common import named_params_and_buffers
    from vime.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor
    from vime.backends.vllm_utils.vllm_engine import VLLMEngine
    from vime.utils.distributed_utils import get_gloo_group
    from vime.utils.tensor_backper import TensorBackuper

    rank = dist.get_rank()
    out = Path(os.environ["VIME_VALIDATION_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    world_size = dist.get_world_size()
    assert world_size % 2 == 0
    engine_count = world_size // 2
    args.actor_num_nodes = 1
    args.actor_num_gpus_per_node = world_size
    args.rollout_num_gpus = world_size
    args.rollout_num_gpus_per_engine = 2
    args.num_gpus_per_node = world_size
    args.colocate = True
    args.resource_layout_spec = None
    if backup is None:
        backup = TensorBackuper.create(lambda: named_params_and_buffers(args, model), single_tag=None)
        backup.backup("actor")
    updater = UpdateWeightFromTensor(
        args, model, lambda: backup.get("actor"), model_name="qwen3_5_moe", quantization_config=None
    )
    if train_step is None:
        model.clear()
    else:
        train_step(1)
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    offloaded_gib = torch.npu.memory_allocated() / 2**30
    assert offloaded_gib < 2, f"Training model retained {offloaded_gib:.2f} GiB"
    print(f"RANK {rank}: CPU backup ready; remaining NPU allocation={offloaded_gib:.3f} GiB", flush=True)
    dist.barrier(group=get_gloo_group())

    @ray.remote(num_cpus=1)
    class AttachedEngine(VLLMEngine):
        def __init__(self, model_path, port):
            super().__init__(SimpleNamespace(hf_checkpoint=model_path), rank=0)
            self.server_host, self.server_port = "127.0.0.1", port

        def state(self):
            return {"version": self._weight_version, "resume_pending": self._resume_pending}

    ports = [28740 + i for i in range(engine_count)]
    servers, server_logs, engines = [], [], []
    report = {
        "rollout_tp": 2,
        "engines": engine_count,
        "training_world_size": world_size,
        "offloaded_npu_gib": offloaded_gib,
        "rounds": [],
    }

    def generate(port):
        base = f"http://127.0.0.1:{port}"
        response = requests.post(
            base + "/v1/completions",
            json={
                "model": "vime-028-validation",
                "prompt": "The capital of France is",
                "max_tokens": 16,
                "temperature": 0,
                "seed": 123,
                "logprobs": 1,
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        assert result["usage"]["completion_tokens"] > 0
        text = result["choices"][0]["text"]
        assert text.strip()
        logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
        assert logprobs and all(value is not None for value in logprobs)
        return {"text": text, "logprobs": logprobs}

    def root_step(action):
        outcome = [None]
        if rank == 0:
            try:
                outcome[0] = {"result": action()}
            except Exception as exc:
                outcome[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(outcome, src=0, group=get_gloo_group())
        assert "error" not in outcome[0], outcome[0]
        return outcome[0]["result"]

    try:

        def launch_servers():
            env = os.environ.copy()
            # A fresh server must not inherit torchrun's trainer rendezvous.
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            env.update(
                VLLM_SERVER_DEV_MODE="1",
                VLLM_ALLOW_INSECURE_SERIALIZATION="1",
                PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
            )
            command = [
                "vllm",
                "serve",
                args.hf_checkpoint,
                "--served-model-name",
                "vime-028-validation",
                "--tensor-parallel-size",
                "2",
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "2",
                "--max-num-batched-tokens",
                "2048",
                "--gpu-memory-utilization",
                "0.72",
                "--enforce-eager",
                "--enable-sleep-mode",
                "--language-model-only",
                "--distributed-executor-backend",
                "mp",
                "--weight-transfer-config",
                '{"backend":"vime_npu_ipc"}',
                "--worker-extension-cls",
                "vime.backends.vllm_utils.npu_weight_transfer.VimeNPUWorkerExtension",
                "--additional-config",
                '{"rl_config":{"enabled":true},"weight_nz_mode":0,"multistream_overlap_shared_expert":false}',
                "--host",
                "127.0.0.1",
                "--port",
                str(ports[0]),
            ]
            devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert len(devices) == world_size
            for index, port in enumerate(ports):
                command[-1] = str(port)
                env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices[2 * index : 2 * index + 2])
                server_log = (out / f"server-{index}.log").open("w")
                server_logs.append(server_log)
                server = subprocess.Popen(
                    command, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
                )
                servers.append(server)
                (out / f"server-{index}.pid").write_text(str(server.pid))
            deadline = time.monotonic() + 600
            pending = set(ports)
            while pending and time.monotonic() < deadline:
                assert all(server.poll() is None for server in servers), "vLLM exited; inspect server logs"
                for port in list(pending):
                    try:
                        if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                            pending.remove(port)
                    except requests.RequestException:
                        pass
                if pending:
                    time.sleep(2)
            assert not pending, f"vLLM startup timed out on ports {pending}"
            return [generate(port) for port in ports]

        report["baseline"] = root_step(launch_servers)
        print(f"RANK {rank}: baseline={report['baseline']!r}", flush=True)
        os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"

        def launch_ray():
            # Ray appends long session/socket names; report directories may exceed AF_UNIX limits.
            ray_dir = tempfile.mkdtemp(prefix="vime028-ray-")
            (out / "ray-path.txt").write_text(ray_dir + "\n")
            context = ray.init(
                address="local",
                num_cpus=max(2, engine_count),
                num_gpus=0,
                resources={"NPU": 0},
                include_dashboard=False,
                _node_ip_address=ray.util.get_node_ip_address(),
                _temp_dir=ray_dir,
                object_store_memory=128 * 1024**2,
            )
            return context.address_info["gcs_address"]

        address = root_step(launch_ray)
        if rank != 0:
            ray.init(address=address)
        engines = root_step(lambda: [AttachedEngine.remote(args.hf_checkpoint, port) for port in ports])
        updater.connect_rollout_engines(
            engines, None, engine_gpu_counts=[2] * engine_count, engine_gpu_offsets=list(range(0, world_size, 2))
        )
        dist.barrier(group=get_gloo_group())
        for version in (1, 2):
            if version == 2 and train_step is not None:
                # Ray installs signal handlers; retain Python frames for native failures.
                faulthandler.enable(all_threads=True)
                train_step(2)
            started = time.monotonic()
            if version == 1:
                root_step(lambda: ray.get([engine.pause_generation.remote() for engine in engines]))
                root_step(lambda: ray.get([engine.release_memory_occupation.remote(level=2) for engine in engines]))
            root_step(
                lambda: ray.get([engine.resume_memory_occupation.remote(tags=["weights"]) for engine in engines])
            )
            updater.update_weights()
            assert updater.weight_version == version
            states = root_step(lambda: ray.get([engine.state.remote() for engine in engines]))
            assert all(state == {"version": str(version), "resume_pending": True} for state in states), states
            if train_step is not None:
                from vime.utils.memory_utils import aggressive_empty_cache
                from vime.utils.reloadable_process_group import destroy_process_groups

                # Match MegatronTrainRayActor.update_weights before rollout KV wake.
                destroy_process_groups()
                aggressive_empty_cache(force_sync=True)
            torch.npu.synchronize()
            torch.npu.empty_cache()
            idle_gib = torch.npu.memory_allocated() / 2**30
            assert idle_gib < 2, f"IPC exports retained {idle_gib:.2f} GiB"
            root_step(
                lambda: ray.get([engine.resume_memory_occupation.remote(tags=["kv_cache"]) for engine in engines])
            )
            output = root_step(lambda: [generate(port) for port in ports])
            assert all(item == output[0] for item in output), "Rollout replicas disagree"
            if train_step is None:
                assert [item["text"] for item in output] == [item["text"] for item in report["baseline"]]
            else:
                previous = report["baseline"] if version == 1 else report["rounds"][-1]["output"]
                assert all(new != old for new, old in zip(output, previous, strict=True)), (
                    "Updated weights had no effect on rollout"
                )
            root_step(lambda: ray.get([engine.pause_generation.remote() for engine in engines]))
            root_step(lambda: ray.get([engine.release_memory_occupation.remote(level=2) for engine in engines]))
            report["rounds"].append(
                {"version": version, "output": output, "idle_npu_gib": idle_gib, "seconds": time.monotonic() - started}
            )
            (out / f"colocate-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"RANK {rank}: ROUND {version} PASS {report['rounds'][-1]}", flush=True)
        print(f"RANK {rank}: PASS two production Megatron-to-vLLM IPC reloads", flush=True)
    finally:
        if rank == 0:
            for engine in engines:
                ray.kill(engine)
        if ray.is_initialized():
            ray.shutdown()
        if rank == 0:
            for server in servers:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
                except ProcessLookupError:
                    pass
        for server_log in server_logs:
            server_log.close()
