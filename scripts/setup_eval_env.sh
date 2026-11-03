#!/usr/bin/env bash
# Creates .venv-eval: an isolated environment for the evaluation client (AgentDojo + its dependencies).
# It is separate from the environment vLLM runs in, so AgentDojo's dependencies (openai, pydantic, anthropic,
# cohere, google-genai, langchain, ...) can never change vLLM's packages. The client only talks to vLLM over HTTP.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), f"Python >= 3.10 needed, found {sys.version}"'
if ! "$PYTHON" -m venv .venv-eval; then
    echo "Could not create a venv. On Ubuntu: sudo apt-get install -y python3-venv (or set PYTHON=/path/to/python3)." >&2
    exit 1
fi
.venv-eval/bin/pip install --quiet --upgrade pip
.venv-eval/bin/pip install --quiet -e ./agentdojo
.venv-eval/bin/python -c "import agentdojo, openai; print('Evaluation environment ready: .venv-eval')"
