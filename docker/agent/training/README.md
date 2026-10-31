# vime A3 release image — one-click build

Target: Ascend A3 (910_93-class) / aarch64, CANN 9.0.0.

Every dependency is cloned from its **official upstream** at a pinned ref and a
small vime patch (`patches/`) is applied — no private forks. See the header of
`Dockerfile.release` for the full baseline table.

---

## 0. Prerequisites you must supply

### (a) Base image — CANN 9.0.0 A3
Default base is `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12` (public; the
`-a3-` tag already ships the ascend910_93 operator kernels, so no separate
A3-ops package is needed). If you cannot pull it, download CANN 9.0.0 from:

    https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0

and build/point `--build-arg BASE_IMAGE=...` at an equivalent CANN 9.0.0 aarch64
base. NOTE: triton-ascend 3.2.1 (below) requires CANN 9.0.0 **B160 or later**.

### (b) triton-ascend wheel (not on public PyPI)
The stack was validated on **triton-ascend 3.2.1**, which is not published on
public PyPI. Download the aarch64 wheel from the official releases page:

    https://gitcode.com/Ascend/triton-ascend/releases

Pick the wheel matching CANN 9.0.0 + py3.12 + aarch64, put it under `wheels/`,
and pass its filename at build time:

    --build-arg TRITON_ASCEND_WHEEL=triton_ascend-3.2.1-<...>-aarch64.whl

A cp311 wheel (`triton_ascend-3.2.1-cp311-...-aarch64.whl`) is already shipped in
`wheels/` and is the default for the cann:900 test build. **cp311 requires a
py3.11 base** (matches cann:900); the py3.12 quay release base needs a **cp312**
wheel — download it from the releases page above and pass its filename.

There is **NO fallback**: 3.2.1 is not on public PyPI, so if neither a matching
wheel (in `wheels/` + `--build-arg TRITON_ASCEND_WHEEL=<file>`) nor a
3.2.1-serving `--build-arg PIP_INDEX_URL=<mirror>` is given, the build errors out
by design.

---

## 1. Build

```
docker build -f Dockerfile.release -t vime:a3-release .
```

With proxy + internal pip mirror + the triton-ascend 3.2.1 wheel in `wheels/`:

```
docker build -f Dockerfile.release -t vime:a3-release \
  --build-arg HTTP_PROXY=$HTTP_PROXY   --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg http_proxy=$http_proxy   --build-arg https_proxy=$https_proxy \
  --build-arg PIP_INDEX_URL=<mirror>   --build-arg PIP_TRUSTED_HOST=<host> \
  --build-arg TRITON_ASCEND_WHEEL=triton_ascend-3.2.1-<...>-aarch64.whl \
  .
```

---

## 2. First run on an NPU host — build the NPU kernels (REQUIRED)

The image build does NOT compile the two hardware/env-dependent packages
(vllm-ascend custom AscendC kernels + fla_npu GDN ops) — their source is cloned
+ patched into the image, but compiling them needs `npu-smi`, the `bisheng`
compiler and a real NPU env that a `docker build` doesn't have. Compile them
ONCE, inside the container, on an Ascend A3 host:

```
docker run -it --name vime \
  --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  vime:a3-release bash

# inside the container:
bash /workspace/build_npu_kernels.sh

# optional — bake the compiled kernels into a final image:
docker commit vime vime:a3-final
```

(Adjust the `--device` / driver mounts to your host.) After this, vllm-ascend
and fla_npu are installed and importable.

## 3. Notes

- **vllm-ascend custom kernels + fla_npu** are compiled by
  `build_npu_kernels.sh` on first run (section 2), NOT during image build.
  fla_npu follows the Ascend slime-ascend qwen3.5-9B tutorial exactly
  (`build.sh --soc=ascend910_93 --pkg --ops=<gdn set>` → install the `.run` →
  `torch_custom/fla_npu/build.sh`); its `install_deps.sh` is skipped (the apt
  packages are installed in the Dockerfile). torch-npu is kept at STOCK 2.10.0
  so the bundled `torchnpugen` matches what fla v26.1.0 expects.
- **vime** is cloned from the public repo `github.com/BryanChen408/vime`
  @ `feature/swe-tasks` — nothing to push. (Override with
  `--build-arg VIME_REPO/VIME_REF`.)
- **Polar / ProRL-Agent-Server** (agentic SWE/math RL server) is NOT included in
  this image; it is a separate concern.

### Local test build (Dockerfile.release.cann900)
`Dockerfile.release.cann900` is identical but its base defaults to the local
image `cann:900` (has CANN 9.0.0 + torch) — for testing the build without
pulling the quay base:

    docker build -f Dockerfile.release.cann900 -t vime:a3-test .

## 4. Baselines (all official upstream + patch)

| repo | upstream | ref | patch |
|------|----------|-----|-------|
| vllm | github.com/vllm-project/vllm | v0.21.0 | patches/vllm-vime.patch |
| vllm-ascend | github.com/vllm-project/vllm-ascend | v0.21.0rc1 | patches/vllm-ascend-vime.patch |
| Megatron-LM | github.com/NVIDIA/Megatron-LM | 3714d81d | patches/megatron-vime.patch |
| MindSpeed | gitcode.com/Ascend/MindSpeed | fc63de5c | patches/mindspeed-vime.patch |
| flash-linear-attention-npu | github.com/flashserve/flash-linear-attention-npu | v26.1.0 | (none) |
| vime | github.com/BryanChen408/vime | feature/swe-tasks | (none) |

## 5. A2 / x86 vLLM 0.28 migration validation

The `migration/vllm-028-colocate` branch is being validated on host 52 in
`vime-028-colocate`, starting from image
`vllm-ascend:a2-cann-9.1.0-vllm-ascend-main-vllm-0.28.0` (`2a0fa6fc2c50`).
This is a separate baseline from the A3 build above. Do not apply the old
vLLM 0.21 patches to this image.

- Python 3.12.13, CPU torch 2.10.0 and torchvision 0.25.0,
  torch-npu 2.10.0.post4, CANN 9.1.0, triton-ascend 3.2.2.
- Image sources: vLLM `2cf0a691`, vllm-ascend `3b7cf0f`. The latter reports
  `0.19.1rc2.dev`; the image tag does not identify an official matched pair
  of 0.28 releases. Preserve these source revisions when reproducing tests.
- Qwen3.6-35B-A3B: training TP1 / PP2 / EP4 / ETP1 / CP1 on eight NPUs;
  four colocated TP2 rollout engines, text only, no MTP or PD.
  The long operator-agent smoke uses CP2 and a 32K per-GPU packing budget.
- VIME upstream baseline `369c813` requires bootstrap mode negotiation.
  Polar `a3-pd` at `e57fa702` and the old host copy at `d7e05242` lack it.
  Matching Polar code is `feat/ascendc-rl-t2a` at `a8e30650`, merged into
  the isolated `migration/vllm-028-smoke` branch at `d89f40ac`.

On 2026-10-30, the matching Polar passed 82 focused tests covering bootstrap,
policy boundaries, inflight tracking, trajectory prefix merging and evaluation
budgets. VIME's actual `prepare_initial_policy` also passed against host 52:
`/migration/028-polar-bootstrap-matched.json`. This checks the handshake;
it does not establish that a complete Polar RL step succeeds.

The production-entry smoke scripts are `tests/run_train028_rl.sh` (short
native F1/GRPO) and `tests/run_train028_polar.sh` (64K operator-agent rollout).
The image's NPU Inductor generated a dynamic-shape weighted SwiGLU wrapper with a constant
kernel extent, and the first formal RL run failed with an SMMU/vector-core
exception (`507035`). The eager path passed changing-token-count forward and
gradient checks against FP32 using `tests/repro_npu_swiglu028.py`.
The fused reproduction passed at 821 tokens but failed at 1033 tokens
(36.7% of output elements outside tolerance). With fusion disabled, a native
`train.py` F1/GRPO step completed on eight NPUs: eight valid samples, gradient
norm 16.2504, optimizer update, weight transfer to all four TP2 engines, and
KV restore. Evidence: `/migration/028-production-rl-no-swiglu/summary.json`
and `completed.json`. This is a single-step check, not long-run convergence
or an end-to-end Polar operator result.
The current scripts retain `TORCHDYNAMO_DISABLE=1` from the existing production
launchers, keeping training helpers eager until NPU dynamic-shape compilation
is validated. The vLLM launcher removes this training variable from its child
environment; explicit Triton kernels are unaffected. With this setting, the
same fused SwiGLU reproduction passes all four token counts and gradient checks
(`/migration/028-swiglu-dynamo-disabled.json`), so the additional
`--no-bias-swiglu-fusion` workaround is no longer needed.
These scripts require a new output directory, eight free NPUs numbered 0–7, the HF
checkpoint, and its converted Megatron checkpoint. Inside this test container:

```bash
source /migration/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_CHECKPOINT=/home/docker/Qwen3.6-35B-A3B
export TRAIN_CHECKPOINT=/home/docker/Qwen3.6-35B-A3B_fused_torch_dist
export VIME_VALIDATION_OUTPUT=/migration/028-production-rl-new
bash tests/run_train028_rl.sh --num-rollout 1
```

The Polar script additionally requires the isolated host-52 services and a free
judge NPU (the test profile uses card 8). Its service launcher and profile are
tracked in the isolated Polar repository under `deploy/ascend_operator/`.
It enables the `qwen3_coder` tool parser and `qwen3` reasoning parser through
the existing launcher environment variables. A CPU-only capture using the real
runtime and CLI measured 36,906 prompt tokens for the first operator, including
18 tool definitions; with 2,048 output tokens the minimum is 38,954, so 32K
cannot admit even the first turn. Evidence: `/migration/028-prompt-capture/`;
the capture script is tracked in Polar at `528897d9`.

The real Polar run at VIME `a2b88b8` completed bootstrap, initial weight sync,
32 agent inference calls and four trajectories of 45,271–56,836 tokens.
All response-token, loss-mask and logprob lengths matched, and all training
logprobs were finite. Each session hit the eight-turn smoke budget before
producing a submission (`submission_missing`, reward zero); this does not
validate successful operator judging. Training then ran out of memory in
MoE recomputation during backward on NPU 4: 1.54 GiB requested with 1.48 GiB
free, 52.22 GiB allocated and 56.19 GiB reserved. Evidence is in
`/migration/028-production-polar-64k-eager/summary.json` and `trajectory-check.json`.
This run used CP1 with expandable segments disabled, unlike the old launcher's
`FEAT_TRAIN_EXPANDABLE=1` setting. This CP1 run did not complete a Polar
training step. Failure notification now bounds long Ray tracebacks to Polar's
2,000-character limit, preserving the exception header and root cause
(`9eda05f`, 38 synchronous bridge tests passed).
Replaying the same four saved trajectories with the existing `forge_load`
rollout entry kept all four colocated vLLM engines live. Restoring training
`expandable_segments:True` still failed in CP1 MoE backward: 1.83 GiB requested,
56.11 GiB active, 56.47 GiB reserved, 1.33 GiB free. Thus allocator fragmentation
alone does not explain the capacity limit. Evidence:
`/migration/028-polar-replay-expandable/summary.json`.
For CP>1, this MindSpeed revision also requires `--optimization-level 2` to
activate its context-parallel feature. A CP2 run with the smoke baseline's
level 0 stopped during model construction with a mock full-attention spec.
A separate import/repatch probe confirmed that level 2 selects
`MindSpeedCPDotProductAttention` (`/migration/028-cp-feature-probe.log`).
The first CP2/level-2 replay also OOMed because the 64K **per-GPU** token
budget packed two trajectories into each microbatch (45.5K–53.9K tokens per
rank). Context length and per-GPU packing budget are different controls:
64K context with CP2 and a 32K per-GPU budget schedules one trajectory per
microbatch (22.6K–28.4K tokens per rank). The existing scheduler reproduced
both schedules with the actual lengths; evidence:
`/migration/028-polar-replay-cp2-opt2/batch-schedule-probe.json`.

The corrected CP2/level-2/32K-budget replay completed one production
`train.py` step at `9eda05f`: all eight ranks finished backward, CPU Adam
completed, and weights plus KV cache were restored on all four TP2 engines.
The four saved samples were unchanged (tokens, masks, logprobs and rewards).
Maximum reported reserved memory was 57.39 GiB; mean train/rollout logprob
absolute difference was 0.05915. Gradient norm was zero, as expected for
these all-zero-reward groups; this is execution validation, not evidence of
learning. Evidence: `/migration/028-polar-replay-cp2-32k/summary.json`,
`completed.json`, `train-metrics.json`, and `replay-check.json`.
The Polar smoke script now selects this configuration. This replay did not
verify the durable Polar epoch transition, and successful operator
submission/judging needs a larger agent budget. It also did not
test generation after training: the API servers exited with Ray shutdown
before the external probe. Tests covered trajectories up to 56,836 tokens;
they do not establish headroom for every 64K workload or multi-step stability.

On 2026-10-31, **two live Polar iterations passed** on physical NPUs 0–7,
using VIME `92658fe` and Polar `319597a4`. Both iterations used the production
`train.py` entry, CP2 / optimization level 2 / 32K per-GPU packing budget,
and four colocated TP2 vLLM engines. Each iteration completed backward on
all eight ranks, CPU Adam, weight transfer, KV restore and a durable policy
commit. Policy epochs advanced 0 -> 1 -> 2; all four engine versions advanced
1 -> 2 -> 3. The second iteration generated fresh agent trajectories carrying
epoch 1, proving online generation after the first training update. The driver
exited 0 and Polar quiesced at epoch 2. Remaining processes in this test's
recorded process group were terminated; NPUs 0–7 were verified idle.

The eight training trajectories contained 44,442–63,037 tokens. Token/mask/
logprob lengths matched and all training logprobs were finite. Maximum reported
reserved memory across all ranks was 57.35 GiB, with no OOM. Mean train/rollout
logprob absolute differences were 0.06994 and 0.05261. All rewards and gradient
norms were zero (`submission_missing` under the eight-turn agent budget), so
this validates execution, not learning or successful operator evaluation.

There were 96 successful model calls across 12 sessions. Two sessions timed
out because the old 600-second session budget includes waiting for a run
worker; VIME rejected one group and successfully collected a replacement in
each iteration. Polar `9da7f59a` increases the future-launch budget to 1,800
seconds. Its YAML semantic/diff checks passed; this live run retained the
600-second setting. One other session's next request exceeded context by one
token (63,489 input + 2,048 requested output > 65,536); its prior valid
trajectory was retained. Longer agent budgets, context-limit handling,
nonzero-reward operator training and long-run stability remain unvalidated.
No generation after the final epoch-2 commit was requested.

Evidence is under `/migration/028-polar-cp2-online-20261031/`: `summary.json`,
`completed.json`, `policy-commits.json`, `final-policy.json`,
`trajectory-check-step{0,1}.json`, `train-metrics.json`,
`rank-memory-and-backward.json`, `all-session-results.json` and `cleanup.json`.
For another two-iteration run, select fresh output and Polar run/control
directories, start the isolated host-52 services, and use the environment
exports above with:

```bash
export VIME_VALIDATION_OUTPUT=/migration/028-polar-cp2-online-new
VALIDATION_TIMEOUT=3600 bash tests/run_train028_polar.sh --num-rollout 2
```

Keep failed policy-control state for diagnosis: a `recovery_required` transition
deliberately blocks a new run. Never reset shared service state to bypass it.
The isolated Polar launcher accepts `POLAR_CONTROL_DIR` (since `319597a4`);
use a fresh directory for an independent run while preserving old evidence.
