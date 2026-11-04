#!/usr/bin/env bash
# Starts the vLLM server with the baseline settings (run it in its own terminal, from the environment where
# vLLM is installed). Same as the command in README.md, plus `tee` to vllm_server.log, which lets the evaluation
# report the model's memory use. Notes:
#   --enable-auto-tool-choice --tool-call-parser hermes   vLLM parses Qwen's tool calls ("native" mode).
#   --kv-cache-dtype fp8                                  part of the baseline configuration; keep it fixed.
#   --max-model-len 6144                                  the largest prompts need about 5,700 tokens; raise
#                                                          to 8192 if the smoke test reports a context error.
# Extra arguments are passed to vLLM, e.g. scripts/serve_vllm.sh --max-model-len 8192
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-vllm_server.log}"

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$MODEL" \
    --max-model-len 6144 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --limit-mm-per-prompt '{"image": 4}' \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "$PORT" \
    "$@" 2>&1 | tee "$SERVER_LOG"
