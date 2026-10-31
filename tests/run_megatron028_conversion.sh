#!/usr/bin/env bash
# Explicit hardware repro; source the training environment before running.
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?Select two free NPUs}"
: "${HF_CHECKPOINT:?Set the HF checkpoint directory}"
: "${VIME_VALIDATION_OUTPUT:?Set the report directory}"
cd "$(dirname "$0")/.."
source scripts/models/qwen3.5-35B-A3B.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export OMP_NUM_THREADS=1
export GLOO_SOCKET_IFNAME=lo
# Match the training launchers: multiple processes on one NPU cannot share the default socket port.
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
# The container hostname is not resolvable: use an explicit local rendezvous.
# Keep normal NPU initialization: MindSpeed sets expert TP metadata inside initialization.
timeout 1200 torchrun --master-addr=127.0.0.1 --master-port="${VALIDATION_MASTER_PORT:-28729}" \
 --nproc-per-node=2 tests/repro_megatron028_conversion.py \
 "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" \
 --qwen-gdn-backend npu --megatron-to-hf-mode raw --tensor-model-parallel-size 2 \
 --expert-tensor-parallel-size 2 --expert-model-parallel-size 1 --pipeline-model-parallel-size 1 \
 --sequence-parallel --micro-batch-size 1 --global-batch-size 1 --seq-length 128 \
 --optimization-level 0 --no-gradient-accumulation-fusion
