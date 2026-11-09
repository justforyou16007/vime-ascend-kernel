from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vime.ray.engine_roles import colocated_prefix_count, resolve_engine_roles
from vime.ray.resource_layout import load_resource_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run-qwen36-35b-polar-multi-pd.sh"
MODEL_SCRIPT = REPO_ROOT / "scripts" / "models" / "qwen3.5-35B-A3B.sh"
SYNC_HYBRID = REPO_ROOT / "scripts" / "start_sync_hybrid.sh"
SYNC_SINGLE52 = REPO_ROOT / "scripts" / "start_sync_hybrid_single52.sh"
SYNC_HOMO_SINGLE52 = REPO_ROOT / "scripts" / "start_sync_homo_single52.sh"
YARN_SMOKE_SINGLE52 = REPO_ROOT / "scripts" / "start_qwen36_yarn_smoke_single52.sh"


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _mode_block() -> str:
    source = _source_text(RUNNER)
    start = source.index('if [ "${FEAT_SYNC_ROLLOUT}" = "1" ]; then')
    end = source.index("\nROLLOUT_ARGS=(", start)
    return source[start:end]


def _gate_block() -> str:
    source = _source_text(RUNNER)
    start = source.index("# ─── 训推模式闸门 ───")
    end = source.index("# ─── polar 数据 / 端点 ───", start)
    return source[start:end]


def _shell_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _evaluate_mode(*, sync: str = "0", tis_disabled: str = "0", factor: str = "1.0") -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
FEAT_SYNC_ROLLOUT={sync}
POLAR_DISABLE_TIS={tis_disabled}
POLAR_SYNC_OVERSUBSCRIBE_FACTOR={factor}
{_mode_block()}
printf 'fn=%s\\n' "$ROLLOUT_FN"
printf 'sched=%s\\n' "${{SCHED_ARGS[*]-}}"
printf 'tis=%s\\n' "${{TIS_ARGS[*]-}}"
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_shell_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _evaluate_gate(
    *,
    sync: str = "0",
    colocate: str = "0",
    train_entry: str = "train_async.py",
    durable: str = "0",
    offload: str = "0",
) -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
MASTER_ADDR=80.48.5.56
ROLLOUT_NODE_IP=80.48.5.64
RESOURCE_LAYOUT=layout.yaml
FEAT_COLOCATE={colocate}
TRAIN_ENTRY={train_entry}
FEAT_OFFLOAD={offload}
FEAT_SYNC_ROLLOUT={sync}
POLAR_POLICY_TRANSITION_ENABLED={durable}
{_gate_block()}
printf 'ok\\n'
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_shell_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _stdout_fields(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _evaluate_yarn_model(*, enabled: str) -> subprocess.CompletedProcess[str]:
    env = _shell_env()
    for name in tuple(env):
        if name == "FEAT_YARN" or name.startswith("YARN_"):
            env.pop(name)
    env["FEAT_YARN"] = enabled
    script = """
set -e
source "$1"
printf 'ARG=%s\n' "${MODEL_ARGS[@]}"
printf 'HF_OVERRIDES=%s\n' "${QWEN36_VLLM_HF_OVERRIDES}"
"""
    return subprocess.run(
        ["bash", "-c", script, "bash", str(MODEL_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_runner_is_shell_valid() -> None:
    result = subprocess.run(["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_save_hf_default_does_not_leak_parameter_expansion_brace() -> None:
    source = _source_text(RUNNER)
    assert "SAVE_HF='/workspace/Qwen3.6-35B-A3B_vime_polar/rollout_{rollout_id}'" in source
    assert '--save-hf "${SAVE_HF}"' in source
    assert '${SAVE_HF:-/workspace/Qwen3.6-35B-A3B_vime_polar/rollout_{rollout_id}}' not in source


def test_async_is_the_default_and_keeps_session_pool() -> None:
    result = _evaluate_mode()
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["fn"] == "vime_bridge.rollout.generate_rollout_polar_async"
    assert fields["sched"] == (
        "--rollout-max-async-level 1 --rollout-scheduler-mode session_pool "
        "--rollout-max-active-sessions 16 --rollout-release-on-postrun"
    )
    assert fields["tis"] == "--use-tis"


def test_sync_is_explicit_and_has_no_async_scheduler_state() -> None:
    result = _evaluate_mode(sync="1")
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["fn"] == "vime_bridge.rollout.generate_rollout_polar_sync"
    assert fields["sched"] == "--rollout-sync-oversubscribe-factor 1.0"
    assert "session_pool" not in fields["sched"]
    assert fields["tis"] == "--use-tis"


@pytest.mark.parametrize("factor", ["1.0", "1.25", "1.5"])
def test_sync_factor_boundaries_are_forwarded(factor: str) -> None:
    result = _evaluate_mode(sync="1", factor=factor)
    assert result.returncode == 0, result.stderr
    assert _stdout_fields(result)["sched"] == f"--rollout-sync-oversubscribe-factor {factor}"


@pytest.mark.parametrize("factor", ["0.99", "1.51", "nan", "inf", "not-a-number"])
def test_sync_factor_out_of_contract_is_rejected(factor: str) -> None:
    result = _evaluate_mode(sync="1", factor=factor)
    assert result.returncode != 0
    assert "POLAR_SYNC_OVERSUBSCRIBE_FACTOR" in result.stderr


def test_tis_can_be_disabled_without_changing_rollout_mode() -> None:
    result = _evaluate_mode(tis_disabled="1")
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["fn"] == "vime_bridge.rollout.generate_rollout_polar_async"
    assert fields["sched"].startswith("--rollout-max-async-level")
    assert fields["tis"] == ""


def test_mode_gate_rejects_sync_with_async_training_entry() -> None:
    result = _evaluate_gate(sync="1", train_entry="train_async.py")
    assert result.returncode != 0
    assert "requires TRAIN_ENTRY=train.py" in result.stderr


def test_mode_gate_rejects_colocate_without_sync() -> None:
    result = _evaluate_gate(colocate="1", train_entry="train.py")
    assert result.returncode != 0
    assert "FEAT_COLOCATE=1 requires FEAT_SYNC_ROLLOUT=1" in result.stderr


def test_durable_gate_requires_sync_train_entry_and_offload() -> None:
    async_entry = _evaluate_gate(durable="1", train_entry="train_async.py")
    assert async_entry.returncode != 0
    assert "durable Polar transitions require TRAIN_ENTRY=train.py" in async_entry.stderr

    without_offload = _evaluate_gate(durable="1", train_entry="train.py", offload="0")
    assert without_offload.returncode != 0
    assert "durable Polar transitions require FEAT_OFFLOAD=1" in without_offload.stderr

    valid = _evaluate_gate(durable="1", train_entry="train.py", offload="1")
    assert valid.returncode == 0, valid.stderr


def test_sync_launcher_pins_sync_durable_and_probe_contract() -> None:
    source = _source_text(SYNC_HYBRID)
    lines = {line.strip() for line in source.splitlines()}
    assert any(line.startswith("FEAT_SYNC_ROLLOUT=1") for line in lines)
    assert any(line.startswith("POLAR_SYNC_OVERSUBSCRIBE_FACTOR=1.0") for line in lines)
    assert any(line.startswith("POLAR_POLICY_TRANSITION_ENABLED=1") for line in lines)
    assert any(line.startswith("VIME_MEM_PROBE=1") for line in lines)
    for async_only in ("POLAR_MAX_ACTIVE_SESSIONS", "POLAR_DRAIN_SESSIONS", "POLAR_MAX_OFF_POLICY_STEPS"):
        assert async_only not in source


def test_single52_launchers_preserve_proven_sync_contracts() -> None:
    hybrid = _source_text(SYNC_SINGLE52)
    homo = _source_text(SYNC_HOMO_SINGLE52)

    for path, source in ((SYNC_SINGLE52, hybrid), (SYNC_HOMO_SINGLE52, homo)):
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "FEAT_SYNC_ROLLOUT=1" in source
        assert "VIME_MEM_PROBE=1" in source
        assert "FEAT_TRAIN_EXPANDABLE=1" in source
        assert "VIME_EMPTY_CACHE_PER_STEP=1" in source
        assert "POLAR_TRAJECTORY_PG_FLOOR=0.05" in source
        assert "HCCL_INTER_HCCS_DISABLE=false" in source
        assert "OPERATOR_DATA_ROOT=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189" in source

    assert "ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15" in hybrid
    assert "NNODES=1  NPUS_PER_NODE=12" in hybrid
    assert "RESOURCE_LAYOUT=/workspace/vime/scripts/resource_layout.single52_hybrid_colocate.yaml" in hybrid
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" in homo
    assert "NNODES=1  NPUS_PER_NODE=16" in homo
    assert "resource_layout.single52_homo_colocate.yaml" in homo


def test_launchers_do_not_reintroduce_online_mtp_and_keep_yarn_opt_in() -> None:
    source = "\n".join(
        _source_text(path) for path in (RUNNER, SYNC_HYBRID, SYNC_SINGLE52, SYNC_HOMO_SINGLE52)
    )
    assert re.search(r"(?i)\bmtp\b", source) is None
    assert "FEAT_YARN=1" not in _source_text(SYNC_HYBRID)
    assert "FEAT_YARN=1" not in _source_text(SYNC_SINGLE52)
    assert "FEAT_YARN=1" not in _source_text(SYNC_HOMO_SINGLE52)


def test_yarn_smoke_profile_is_isolated_and_capacity_matched() -> None:
    source = _source_text(YARN_SMOKE_SINGLE52)

    assert "export FEAT_YARN=1" in source
    assert 'YARN_SMOKE_CONTEXT_LEN="${YARN_SMOKE_CONTEXT_LEN:-262144}"' in source
    assert '"${YARN_SMOKE_CONTEXT_LEN}" -ne 262144' in source
    assert "homo stage-1 launcher only supports YARN_SMOKE_CONTEXT_LEN=262144" in source
    for name in (
        "SEQ_LENGTH",
        "MAX_POSITION_EMBEDDINGS",
        "ROLLOUT_MAX_CONTEXT_LEN",
        "VLLM_MAX_MODEL_LEN",
    ):
        assert f'export {name}="${{YARN_SMOKE_CONTEXT_LEN}}"' in source
    assert "Qwen3.6-35B-A3B_yarn_smoke" in source
    assert 'export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"' in source
    assert "YARN_PREFLIGHT_ONLY" in source
    assert 'export VIME_SAVE_TIS_LOGPROBS="${VIME_SAVE_TIS_LOGPROBS:-${POLAR_OUTPUT_DIR}/tis_evidence}"' in source
    assert 'exec bash "${SCRIPT_DIR}/start_sync_homo_single52.sh"' in source
    assert "start_sync_hybrid_single52.sh" not in source
    assert "profile.t2a.yaml" not in source
    assert "profile.vime.yaml" not in source


def test_yarn_smoke_profile_is_shell_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(YARN_SMOKE_SINGLE52)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_yarn_stage1_smoke_rejects_a_later_capacity_length() -> None:
    env = _shell_env()
    env["YARN_SMOKE_CONTEXT_LEN"] = "270000"
    result = subprocess.run(
        ["bash", str(YARN_SMOKE_SINGLE52)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "homo stage-1 launcher only supports YARN_SMOKE_CONTEXT_LEN=262144" in result.stderr


def test_qwen36_model_defaults_to_standard_rope() -> None:
    result = _evaluate_yarn_model(enabled="0")
    assert result.returncode == 0, result.stderr
    args = [line.removeprefix("ARG=") for line in result.stdout.splitlines() if line.startswith("ARG=")]
    overrides = json.loads(_stdout_fields(result)["HF_OVERRIDES"])

    assert args[args.index("--position-embedding-type") + 1] == "rope"
    assert args[args.index("--rotary-percent") + 1] == "0.25"
    assert not any("yarn" in arg.lower() for arg in args)
    assert overrides == {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}


def test_qwen36_yarn_uses_one_fingerprint_for_training_and_rollout() -> None:
    result = _evaluate_yarn_model(enabled="1")
    assert result.returncode == 0, result.stderr
    args = [line.removeprefix("ARG=") for line in result.stdout.splitlines() if line.startswith("ARG=")]
    overrides = json.loads(_stdout_fields(result)["HF_OVERRIDES"])
    rope = overrides["text_config"]["rope_parameters"]

    assert args[args.index("--position-embedding-type") + 1] == rope["rope_type"] == "yarn"
    assert float(args[args.index("--rotary-base") + 1]) == rope["rope_theta"] == 10000000
    assert float(args[args.index("--rotary-percent") + 1]) == rope["partial_rotary_factor"] == 0.25
    assert float(args[args.index("--rotary-scaling-factor") + 1]) == rope["factor"] == 4.0
    assert (
        int(args[args.index("--yarn-original-max-position-embeddings") + 1])
        == rope["original_max_position_embeddings"]
        == 262144
    )
    assert rope["truncate"] is True
    assert "--yarn-correction-range-round-to-int" in args
    assert '--vllm-hf-overrides "${QWEN36_VLLM_HF_OVERRIDES}"' in _source_text(RUNNER)
    assert "--vllm-allow-long-max-model-len" in _source_text(RUNNER)
    assert '--max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-${SEQ_LENGTH:-131072}}"' in _source_text(RUNNER)


def test_qwen36_yarn_rejects_invalid_boolean_gate() -> None:
    result = _evaluate_yarn_model(enabled="invalid")

    assert result.returncode != 0
    assert "FEAT_YARN must be 0 or 1" in result.stderr


@pytest.mark.parametrize(
    ("filename", "actor", "rollout", "per_engine", "shared", "dedicated", "prefix"),
    [
        ("resource_layout.dual56train57infer_pd.yaml", 16, 8, 2, 0, 8, 0),
        ("resource_layout.hybrid56cola64infer.yaml", 16, 24, 4, 16, 8, 4),
        ("resource_layout.single52_hybrid_colocate.yaml", 8, 12, 2, 8, 4, 4),
        ("resource_layout.single52_homo_colocate.yaml", 16, 12, 2, 12, 0, 6),
    ],
)
def test_resource_layout_topology_fingerprint(
    filename: str,
    actor: int,
    rollout: int,
    per_engine: int,
    shared: int,
    dedicated: int,
    prefix: int,
) -> None:
    layout = load_resource_layout(REPO_ROOT / "scripts" / filename)
    assert layout.actor_num_gpus == actor
    assert layout.rollout_num_gpus == rollout
    assert layout.rollout_num_gpus_per_engine == per_engine
    assert layout.rollout_shared_num_gpus == shared
    assert layout.rollout_dedicated_num_gpus == dedicated

    args = type(
        "Args",
        (),
        {"resource_layout_spec": layout, "rollout_num_gpus_per_engine": per_engine},
    )()
    roles = resolve_engine_roles(args)
    assert len(roles) == rollout // per_engine
    assert colocated_prefix_count(roles) == prefix


# ─── Polar on a separate host (training Pod ⇄ Polar host ⇄ SSH ⇄ NPU VMs) ───


def _polar_endpoint_block() -> str:
    source = _source_text(RUNNER)
    start = source.index("# ─── polar 数据 / 端点 ───")
    end = source.index("\n", source.index("VLLM_ROUTER_PORT=", start))
    return source[start:end]


def _no_proxy_lines() -> str:
    lines = [
        line
        for line in _source_text(RUNNER).splitlines()
        if line.startswith(("export no_proxy=", "export NO_PROXY="))
    ]
    assert len(lines) == 2, lines
    return "\n".join(lines)


def _polar_preflight_function() -> str:
    source = _source_text(RUNNER)
    start = source.index("polar_preflight() {")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


def _clean_proxy_env(**overrides: str) -> dict[str, str]:
    env = _shell_env()
    for name in tuple(env):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"} or name.startswith("POLAR_"):
            env.pop(name)
    env.update(overrides)
    return env


def _resolve_polar_endpoint(**env_overrides: str) -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
MASTER_ADDR=80.48.5.56
{_polar_endpoint_block()}
printf 'url=%s\\n' "$POLAR_ROLLOUT_URL"
printf 'host=%s\\n' "$POLAR_HOST"
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_clean_proxy_env(**env_overrides),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("overrides", "url", "host"),
    [
        ({}, "http://80.48.5.56:8080", "80.48.5.56"),
        ({"POLAR_HOST": "10.20.30.40", "POLAR_ROLLOUT_PORT": "8180"}, "http://10.20.30.40:8180", "10.20.30.40"),
        ({"POLAR_ROLLOUT_URL": "http://polar-box.local:9000/"}, "http://polar-box.local:9000", "polar-box.local"),
        ({"POLAR_HOST": "1.1.1.1", "POLAR_ROLLOUT_URL": "http://10.0.0.9:8180"}, "http://10.0.0.9:8180", "10.0.0.9"),
    ],
)
def test_polar_endpoint_can_live_on_a_separate_host(overrides: dict[str, str], url: str, host: str) -> None:
    result = _resolve_polar_endpoint(**overrides)
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["url"] == url
    assert fields["host"] == host


@pytest.mark.parametrize("bad_url", ["10.0.0.9:8180", "http://10.0.0.9:80x", "ftp://10.0.0.9:21"])
def test_polar_endpoint_rejects_malformed_url(bad_url: str) -> None:
    result = _resolve_polar_endpoint(POLAR_ROLLOUT_URL=bad_url)
    assert result.returncode != 0
    assert "POLAR_ROLLOUT_URL must look like" in result.stderr


def test_polar_host_reaches_no_proxy_before_ray_starts() -> None:
    source = _source_text(RUNNER)
    resolved = source.index("POLAR_HOST=$(python3")
    exported = source.index("export no_proxy=")
    # Both raylets (head and worker) must inherit the Polar no_proxy entry.
    assert resolved < exported < source.index("ray start --head") < source.index("ray start --address")
    assert "${POLAR_HOST}" in source[exported : source.index("\n", exported)]


def test_polar_preflight_runs_on_head_before_ray_teardown_only() -> None:
    source = _source_text(RUNNER)
    head = source.index('if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then')
    worker = source.index("\nelse\n   # worker", head)
    assert head < source.index("   polar_preflight\n", head) < source.index("ray stop --force", head) < worker
    assert "polar_preflight" not in source[worker:]


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        status = 200 if self.path == "/health" else 404
        body = b'{"status":"ok"}' if status == 200 else b"missing"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base signature
        pass  # keep pytest output free of per-request access logs


def _serve_health(host: str) -> ThreadingHTTPServer:
    try:
        server = ThreadingHTTPServer((host, 0), _HealthHandler)
    except OSError as exc:
        pytest.skip(f"cannot bind {host}: {exc}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_launcher_polar_path(**env_overrides: str) -> subprocess.CompletedProcess[str]:
    """Endpoint resolution + the real no_proxy export + preflight, as the head runs them."""
    script = f"""
set -e
MASTER_ADDR=80.48.5.56
CURRENT_IP=80.48.5.56
ROLLOUT_NODE_IP=80.48.5.64
VLLM_ROUTER_IP=80.48.5.56
{_polar_endpoint_block()}
{_no_proxy_lines()}
{_polar_preflight_function()}
polar_preflight
printf 'after_preflight=1\\n'
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_clean_proxy_env(POLAR_PREFLIGHT_WAIT="0", **env_overrides),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_remote_polar_bypasses_http_proxy_and_passes_preflight() -> None:
    # 127.0.0.2 is outside the launcher's fixed no_proxy entries, so the probe only
    # avoids the dead proxy if the launcher really appended the Polar host.
    server = _serve_health("127.0.0.2")
    dead_proxy = f"http://127.0.0.1:{_unused_port()}"
    try:
        result = _run_launcher_polar_path(
            POLAR_ROLLOUT_URL=f"http://127.0.0.2:{server.server_address[1]}",
            http_proxy=dead_proxy,
            HTTP_PROXY=dead_proxy,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 0, result.stderr
    assert "[polar] preflight OK" in result.stdout
    assert "sglang_router_url must be http://80.48.5.56:8001" in result.stdout
    assert _stdout_fields(result)["after_preflight"] == "1"


def test_polar_preflight_fails_fast_when_polar_is_unreachable() -> None:
    result = _run_launcher_polar_path(POLAR_ROLLOUT_URL=f"http://127.0.0.1:{_unused_port()}")
    assert result.returncode != 0
    assert "cannot reach the Polar rollout server" in result.stderr
    assert "after_preflight" not in result.stdout


def test_polar_preflight_rejects_a_non_success_health_status() -> None:
    server = _serve_health("127.0.0.1")
    try:
        result = _run_launcher_polar_path(
            POLAR_ROLLOUT_URL=f"http://127.0.0.1:{server.server_address[1]}/missing-prefix"
        )
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode != 0
    assert "HTTP 404" in result.stderr


def test_polar_preflight_can_be_skipped() -> None:
    result = _run_launcher_polar_path(
        POLAR_ROLLOUT_URL=f"http://127.0.0.1:{_unused_port()}",
        POLAR_PREFLIGHT="0",
    )
    assert result.returncode == 0, result.stderr
    assert "preflight skipped" in result.stdout


def _run_sync_hybrid_with_stub_runner(tmp_path: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    """Execute start_sync_hybrid.sh but replace the nested `bash <runner>` with a recorder."""
    real_bash = shutil.which("bash")
    assert real_bash is not None
    stub = tmp_path / "bash"
    stub.write_text('#!/bin/sh\nprintf "runner=%s\\n" "$*"\nprintf "url=%s\\n" "$POLAR_ROLLOUT_URL"\n')
    stub.chmod(0o755)
    env = _clean_proxy_env(**env_overrides)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        [real_bash, str(SYNC_HYBRID)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_sync_hybrid_requires_explicit_polar_host(tmp_path: Path) -> None:
    missing = _run_sync_hybrid_with_stub_runner(tmp_path)
    assert missing.returncode != 0
    assert "POLAR_HOST" in missing.stderr
    assert "runner=" not in missing.stdout

    given = _run_sync_hybrid_with_stub_runner(tmp_path, POLAR_HOST="10.20.30.40")
    assert given.returncode == 0, given.stderr
    fields = _stdout_fields(given)
    assert fields["runner"] == "scripts/run-qwen36-35b-polar-multi-pd.sh"
    assert fields["url"] == "http://10.20.30.40:8180"


def test_hybrid_layout_reserves_no_training_npu_for_remote_polar() -> None:
    layout = load_resource_layout(REPO_ROOT / "scripts" / "resource_layout.hybrid56cola64infer.yaml")
    assert layout.polar_reserved == ()
