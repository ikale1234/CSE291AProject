#!/usr/bin/env bash
# Recommended way to start the vLLM server (run it in its own terminal or tmux pane, from the environment
# where vLLM is installed). Same as the plain command, plus:
#   --enable-auto-tool-choice --tool-call-parser hermes   vLLM parses Qwen's tool calls ("native" mode).
#     Without them the evaluation still works: it parses the tool calls from the text itself ("text" mode).
#   tee to vllm_server.log                                lets the evaluation report the model's memory use.
# Extra arguments are passed to vLLM, e.g. scripts/serve_vllm.sh --gpu-memory-utilization 0.85
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-vllm_server.log}"

vllm serve "$MODEL" \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 4}' \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "$PORT" \
    "$@" 2>&1 | tee "$SERVER_LOG"
