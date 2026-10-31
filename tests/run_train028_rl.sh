#!/usr/bin/env bash
# Explicit eight-NPU production-entry smoke test; source /migration/env.sh first.
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?Select eight free NPUs}"
: "${HF_CHECKPOINT:?Set the HF checkpoint directory}"
: "${TRAIN_CHECKPOINT:?Set the matching converted Megatron checkpoint}"
: "${VIME_VALIDATION_OUTPUT:?Set a new report directory}"
cd "$(dirname "$0")/.."
source scripts/models/qwen3.5-35B-A3B.sh
CHAT_TEMPLATE_ARGS=(--apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}')
if [[ "${VIME_VALIDATE_POLAR:-0}" == 1 ]]; then
    # Polar applies its own chat template to the original task instruction.
    CHAT_TEMPLATE_ARGS=()
fi
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo
export TORCHDYNAMO_DISABLE=1
export QWEN36_CAUSAL_CONV1D_IMPL=triton VIME_MEM_PROBE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60255 HCCL_NPU_SOCKET_PORT_RANGE=61000-61255
export no_proxy="127.0.0.1,localhost,80.48.5.52" NO_PROXY="127.0.0.1,localhost,80.48.5.52"
unset RAY_ADDRESS
mkdir "$VIME_VALIDATION_OUTPUT"
python - <<'PY'
import json, os
from pathlib import Path
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',') == list(map(str, range(8)))
assert (Path(os.environ['TRAIN_CHECKPOINT']) / 'latest_checkpointed_iteration.txt').is_file()
rows = [
    {'text': 'Name three European capital cities. Answer in one short sentence.', 'label': 'Paris London Berlin'},
    {'text': 'Name three planets in the Solar System. Answer in one short sentence.', 'label': 'Mercury Venus Earth'},
]
(Path(os.environ['VIME_VALIDATION_OUTPUT']) / 'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
PY
timeout "${VALIDATION_TIMEOUT:-1800}" python - "${MODEL_ARGS[@]}" \
 --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$TRAIN_CHECKPOINT" \
 --megatron-to-hf-mode raw --qwen-gdn-backend npu --optimization-level 0 \
 --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 \
 --rollout-num-gpus-per-engine 2 --colocate \
 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
 --expert-tensor-parallel-size 1 --expert-model-parallel-size 4 --context-parallel-size 1 \
 --micro-batch-size 1 --global-batch-size 8 --seq-length 256 \
 --use-dynamic-batch-size --max-tokens-per-gpu 256 \
 --no-gradient-accumulation-fusion --accumulate-allreduce-grads-in-fp32 \
 --attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 \
 --attention-backend flash --use-flash-attn \
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 --optimizer adam --lr 3e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
 --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
 --prompt-data "$VIME_VALIDATION_OUTPUT/prompts.jsonl" --input-key text --label-key label \
 "${CHAT_TEMPLATE_ARGS[@]}" \
 --num-rollout 2 --rollout-batch-size 2 --n-samples-per-prompt 4 \
 --rollout-max-response-len 64 --rollout-max-context-len 256 --rollout-temperature 0.9 \
 --rm-type f1 --advantage-estimator grpo --eps-clip 0.2 \
 --save-debug-rollout-data "$VIME_VALIDATION_OUTPUT/rollout_{rollout_id}.pt" \
 --vllm-router-ip 127.0.0.1 --vllm-router-port 28780 --rollout-lb-proxy \
 --vllm-max-model-len 2048 --vllm-max-num-batched-tokens 2048 --vllm-max-num-seqs 4 \
 --vllm-gpu-memory-utilization 0.72 --vllm-enforce-eager --vllm-language-model-only \
 --vllm-additional-config '{"rl_config":{"enabled":true},"weight_nz_mode":0,"multistream_overlap_shared_expert":false}' \
 --update-weight-buffer-size 134217728 "$@" <<'PY'
import faulthandler, json, os, runpy, tempfile
from pathlib import Path
import torch_npu
import mindspeed.megatron_adaptor
import ray
out = Path(os.environ['VIME_VALIDATION_OUTPUT'])
context = ray.init(address='local', num_cpus=32, num_gpus=0, resources={'NPU': 8},
                   include_dashboard=False, _temp_dir=tempfile.mkdtemp(prefix='v028rl-'))
(out / 'ray-address.json').write_text(json.dumps(dict(context.address_info), default=str))
faulthandler.enable()
try:
    result = runpy.run_path('train.py', run_name='__main__')
    args = result['args']
    assert all((out / f'rollout_{i}.pt').is_file() for i in range(args.num_rollout))
    (out / 'completed.json').write_text(json.dumps({
        'rollouts': args.num_rollout, 'entry': 'train.py', 'rollout_function': args.rollout_function_path,
    }))
finally:
    ray.shutdown()
PY
