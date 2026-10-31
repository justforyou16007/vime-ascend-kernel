#!/usr/bin/env bash
# Explicit hardware test; source /migration/env.sh and reserve eight NPUs first.
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?Select eight free NPUs}"
: "${HF_CHECKPOINT:?Set the HF checkpoint directory}"
: "${VIME_VALIDATION_OUTPUT:?Set the report directory}"
cd "$(dirname "$0")/.."
source scripts/models/qwen3.5-35B-A3B.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
timeout 1800 torchrun --master-addr=127.0.0.1 --master-port="${VALIDATION_MASTER_PORT:-28729}" \
 --nproc-per-node=8 tests/repro_megatron028_training.py \
 "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" \
 --qwen-gdn-backend npu --megatron-to-hf-mode raw --tensor-model-parallel-size 1 \
 --expert-tensor-parallel-size 1 --expert-model-parallel-size 4 --pipeline-model-parallel-size 2 \
 --context-parallel-size 1 --micro-batch-size 1 --global-batch-size 8 --seq-length 128 \
 --optimization-level 0 --no-gradient-accumulation-fusion --accumulate-allreduce-grads-in-fp32 \
 --attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 \
 --attention-backend flash --use-flash-attn \
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 --optimizer adam --lr 3e-6 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
 --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
