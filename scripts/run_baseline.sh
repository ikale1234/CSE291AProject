#!/usr/bin/env bash
# Phase 1 baseline in one command. Uses the vLLM server already running on localhost:$PORT (default 8000).
#   1. creates the evaluation environment (.venv-eval) if it does not exist
#   2. waits for the server, then runs a 10-request smoke test (stops here if it fails)
#   3. runs the full evaluation: 269 pairs x 2 sides x 3 repeats + a sequential latency pass
# Results: vpi_harness/runs/<RUN_NAME>/summary.md (also printed at the end).
#
# Environment variables: PORT (8000), BASE_URL, RUN_NAME, SERVER_LOG (path to the vLLM log, for memory stats).
# Extra arguments go to run_eval.py, e.g.:  scripts/run_baseline.sh --repeats 1 --concurrency 8
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE_URL="${BASE_URL:-http://localhost:${PORT:-8000}/v1}"
RUN_NAME="${RUN_NAME:-baseline_qwen3vl8b_bf16_$(date +%Y%m%d_%H%M%S)}"
if [ -z "${SERVER_LOG:-}" ] && [ -f vllm_server.log ]; then SERVER_LOG=vllm_server.log; fi

[ -x .venv-eval/bin/python ] || bash scripts/setup_eval_env.sh
PY=.venv-eval/bin/python

echo "Waiting for the vLLM server at $BASE_URL (model download and loading can take several minutes) ..."
for i in $(seq 1 180); do
    if curl -sf "$BASE_URL/models" > /dev/null; then break; fi
    if [ "$i" -eq 180 ]; then echo "No server at $BASE_URL after 15 minutes." >&2; exit 1; fi
    sleep 5
done

echo "== Smoke test"
"$PY" vpi_harness/run_eval.py --base-url "$BASE_URL" --smoke --run-name "${RUN_NAME}_smoke"

echo "== Full evaluation: $RUN_NAME"
"$PY" vpi_harness/run_eval.py --base-url "$BASE_URL" --run-name "$RUN_NAME" \
    ${SERVER_LOG:+--server-log "$SERVER_LOG"} "$@"
echo "Done. Results in vpi_harness/runs/$RUN_NAME/"
