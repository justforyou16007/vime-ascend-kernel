#!/usr/bin/env bash
# Requires the isolated host-52 Polar profile.vime028-smoke.yaml services.
set -euo pipefail
cd "$(dirname "$0")/.."
export VIME_VALIDATE_POLAR=1
export VLLM_TOOL_CALL_PARSER=qwen3_coder VLLM_REASONING_PARSER=qwen3
exec bash tests/run_train028_rl.sh \
 --rollout-function-path vime_bridge.rollout.generate_rollout_polar_sync \
 --eval-function-path vime_bridge.rollout.generate_rollout_polar_sync \
 --prompt-data /home/docker/datasets/ascendc-rl-datasets/NPUKernelBench/operator_tasks.npukernelbench.jsonl \
 --input-key prompt --metadata-key metadata --reward-key score \
 --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards \
 --polar-url http://80.48.5.52:8080 --polar-run-id "$(basename "$VIME_VALIDATION_OUTPUT")" \
 --polar-reward-key score --polar-task-id-template '{args.polar_run_id}-op-{rollout_id}-{sample.group_index}' \
 --operator-tasks-dir /home/docker/datasets/ascendc-rl-datasets/NPUKernelBench/src_simple \
 --polar-policy-transition-enabled --rollout-request-timeout 900 \
 --rollout-min-complete-accept-fraction 1 --rollout-sync-oversubscribe-factor 1 \
 --rollout-batch-size 2 --n-samples-per-prompt 2 --global-batch-size 4 --use-dynamic-global-batch-size \
 --context-parallel-size 2 --optimization-level 2 \
 --train-env-vars '{"PYTORCH_NPU_ALLOC_CONF":"expandable_segments:True"}' \
 --seq-length 65536 --rollout-max-context-len 65536 --rollout-max-response-len 2048 \
 --max-tokens-per-gpu 32768 --chunked-lm-head --log-probs-chunk-size 64 \
 --vllm-max-model-len 65536 --vllm-max-num-seqs 2 --use-tis "$@"
