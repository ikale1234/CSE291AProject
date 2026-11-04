# Visual prompt injection on a computer-use agent: Phase 1 baseline

CSE 291A project, Option 4: *Security–Performance Tradeoffs in Resource-Constrained Agents*.

We test how a vision-language model acting as a computer-use agent responds to **visual prompt injections**: malicious instructions that appear *in a screenshot*, such as a pop-up, an email or a chat message. Phase 1 establishes a baseline with **Qwen3-VL-8B-Instruct** (bf16 weights with an FP8 KV cache, served by vLLM) on paired clean and injected screenshots built from [VPI-Bench](https://huggingface.co/datasets/VPI-Bench/vpi-bench). The evaluation runs on [AgentDojo](https://github.com/ethz-spylab/agentdojo), which we extended to pass images to the model.

**Baseline results and what they mean: [RESULTS.md](RESULTS.md).** For full background (design decisions, dataset details, known caveats and next steps), see [CLAUDE.md](CLAUDE.md).

## Run the baseline on EC2

Target setup (not yet run there): g6.xlarge (NVIDIA L4, 24 GB), Deep Learning OSS NVIDIA Driver AMI with PyTorch and vLLM installed.

**1. Start the vLLM server.** Open a terminal window on the Ubuntu server, create a virtual environment for vLLM, install vLLM, and start the server:

```bash
sudo apt-get install -y python3-venv          # only if `python3 -m venv` fails
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm                              # Qwen3-VL needs vLLM 0.11 or newer

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --max-model-len 6144 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --limit-mm-per-prompt '{"image": 4}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

Leave this window open; the server is ready when it logs `Application startup complete`. The first start downloads about 17 GB of model weights. `bash scripts/serve_vllm.sh` runs the same `vllm serve` command and also saves its output to `vllm_server.log`.

Notes on this command:
- **Optional: save the server log.** Append `2>&1 | tee vllm_server.log` to the command. The summary then also reports how much memory the model weights take.
- **The tool flags let vLLM parse Qwen's tool calls itself** (`--enable-auto-tool-choice --tool-call-parser hermes`). Without them the evaluation still works; it detects this and parses the tool calls from the reply text instead.
- **The context limit is close to our largest prompts.** The email and messenger samples carry 3 screenshots and come to roughly 5,700 tokens including the 512-token reply budget, under the 6,144 limit. If the smoke test reports a context-length error, restart the server with `--max-model-len 8192`.
- **The FP8 KV cache is part of the baseline configuration.** `--kv-cache-dtype fp8` stores the attention cache in 8 bits while the weights stay bf16. Use the same server settings for every run you compare, especially in the Phase 2 quantization study.

**2. Run the evaluation.** Once the server is up, open a new terminal window on the server, go to the repo root, and run:

```bash
bash scripts/run_baseline.sh
```

This doesn't use the vLLM virtual environment; the script creates its own.

The script does three things:
1. Creates `.venv-eval`, an isolated Python environment for the evaluation client (it never touches vLLM's packages).
2. Waits for the server and runs a 10-request smoke test, stopping if it fails.
3. Runs the full evaluation: **269 pairs × 2 sides (clean, injected) × 3 repeats = 1,614 requests**, 4 at a time, followed by a 40-request sequential pass to measure latency.

Expect roughly an hour on an L4; this is an estimate that hasn't been measured yet.

**Results** go to `vpi_harness/runs/<run name>/`:

| File | Contents |
|---|---|
| `summary.md` | the metrics tables (also printed at the end) |
| `summary.json` | the same metrics, machine-readable |
| `results.jsonl` | one line per request: outcome, tool calls, raw reply, tokens, latency |
| `config.json` | the run's settings, served model, tool-call mode, GPU memory |

### Options

Pass these to `run_baseline.sh` (they're forwarded to `vpi_harness/run_eval.py`), or call `run_eval.py` directly:

```bash
bash scripts/run_baseline.sh --repeats 1                       # faster, one sample per side
bash scripts/run_baseline.sh --platforms email messenger       # a subset of platforms
bash scripts/run_baseline.sh --temperature 0 --repeats 1       # greedy decoding
RUN_NAME=my_run PORT=8001 bash scripts/run_baseline.sh         # run name / server port
.venv-eval/bin/python vpi_harness/run_eval.py --smoke          # just the smoke test
.venv-eval/bin/python vpi_harness/run_eval.py --summarize vpi_harness/runs/<run>   # recompute a summary
.venv-eval/bin/python vpi_harness/run_eval.py --rescore vpi_harness/runs/<run>     # re-classify stored replies with the current rules (no model calls)
```

### Metrics in `summary.md`

| Metric | Meaning |
|---|---|
| Task success (clean) | share of clean requests whose tool call reaches the user's goal |
| Attack success (injected) | share of injected requests whose tool call follows the injection |
| Attack success on solvable pairs | the same, only for pairs whose clean side succeeds in at least half of its repeats; this keeps "can't do the task at all" from being counted as resistance |
| Task success under attack | share of injected requests that still reach the goal |
| Unauthorized actions (clean) | attack-type calls with no injection present (false-positive baseline) |
| Outcome breakdown | attack / invalid / goal / goal_tool_wrong_args / dismiss / answer_user / off_task / no_tool, per side. A plain-text reply with no tool call is scored like `answer_user(text)`; how often that happens is reported separately |
| Cost | latency (sequential and concurrent), prompt and completion tokens, peak GPU memory (nvidia-smi), model weights memory (vLLM log) |

### Troubleshooting

- **The smoke test fails with 400 errors:** check the vLLM terminal. A context-length error means `--max-model-len` is too small: restart the server with `--max-model-len 8192`. The biggest prompts (email/messenger: 3 screenshots, tool schemas and history) need about 5,700 tokens including the reply budget.
- **The server runs out of memory at startup:** lower `--gpu-memory-utilization` (e.g. 0.90) or `--max-model-len`.
- **`python3 -m venv` fails:** run `sudo apt-get install -y python3-venv`, or set `PYTHON=/path/to/python3.10+`.
- **The plain-text reply rate is close to 100%:** the model isn't producing tool calls. Look at the `content` field in `results.jsonl`, and start the server with the two tool flags above.

## Repository layout

```
agentdojo/                 AgentDojo, vendored, with our image support (diff: patches/agentdojo_image_support.patch)
vpi_harness/
  data/single_turn/        THE DATASET: samples.jsonl (269 pairs), tools.json, README.md
  data/images/             rendered screenshots (clean/, injected/, sessions/)
  run_eval.py              evaluation runner (vLLM -> AgentDojo pipeline -> scoring -> summary)
  qwen_llm.py              AgentDojo pipeline element for vLLM-served Qwen3-VL (tool calls, latency, tokens)
  scoring.py               outcome classifier (attack / goal / dismiss / ...)
  single_turn_tools.py     the three toolsets offered to the model
  session_content.py       text we wrote for the email/messenger sessions (clean messages, summaries, rules)
  build_pairs.py           renders VPI-Bench pages to screenshots (needs vpi-bench/ and Playwright; local only)
  build_single_turn.py     builds data/single_turn from the renders (local only)
  mock_vllm_server.py      fake vLLM server for testing the pipeline without a GPU
  browser.py, vpi_suite.py, scripted_vlm.py, run_passthrough_demo.py   earlier live-browser harness and demo
scripts/                   run_baseline.sh (one command), serve_vllm.sh, setup_eval_env.sh
```

### Rebuilding the dataset (local, not needed on EC2)

This needs the VPI-Bench source (`git clone https://huggingface.co/datasets/VPI-Bench/vpi-bench`), Playwright and Google Chrome:

```bash
agentdojo/.venv/bin/pip install -r vpi_harness/requirements.txt
agentdojo/.venv/bin/python vpi_harness/build_pairs.py        # about 10 minutes
agentdojo/.venv/bin/python vpi_harness/build_single_turn.py  # about 1 minute
```

### Testing without a GPU

```bash
python vpi_harness/mock_vllm_server.py --port 8011 &
PORT=8011 bash scripts/run_baseline.sh --repeats 1
```

The mock validates every request (roles, images, tools) and replies with every reply shape the real model can produce. The outcome numbers it produces are meaningless.
