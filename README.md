# LLM Serving Bench

[![CI](https://github.com/lucioduran/llm-serving-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/llm-serving-bench/actions/workflows/ci.yml)

## Problem

Serving claims need token-level timing, actual hardware, comparable requests and
visible failures. Text chunks alone do not establish time per model token.

## Scope

A small, auditable CPU experiment compares one-thread and two-thread PyTorch
serving of SmolLM2-135M-Instruct. Each configuration serves identical recorded
requests through a real loopback HTTP SSE connection. Client concurrency is 1 or
2; input length is exactly 64 or 256 model tokens. One generation slot queues
requests. This measures a reference implementation, not a production server contest.

## Quick start

Python 3.11 is required. Offline report verification has no runtime dependencies:

```sh
git clone https://github.com/lucioduran/llm-serving-bench.git
cd llm-serving-bench
python -m serving_bench verify results/reference/report.json
```

## Usage

For model execution, create a project-local environment and install the reference
packages. The published run uses Intel macOS, Python 3.11.13 and PyTorch 2.2.2.
Other platforms may need an appropriate PyTorch wheel; record any changed version.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-models.txt
python -m serving_bench download
python -m serving_bench run --output work/my-run --hourly-usd 0.10
python -m serving_bench verify work/my-run/report.json
```

Use `.venv\Scripts\Activate.ps1` on Windows. The output directory must not exist.
Download fetches pinned public model files without an account or paid API.
Inference then loads only local safetensors with remote code disabled. The CLI
returns 0 on success and 2 on invalid input or execution failure. A completed run
can contain failed requests: inspect each cell's error count. Interrupted runs
retain completed cells; verification rejects an incomplete matrix.

## Architecture

```text
Recorded prompts -> closed-loop HTTP clients -> single generation slot
                            |                         |
                     token arrival times       FP32 CPU model
                            |                         |
                     raw report <--------- token-ID SSE events
                            |
                  offline metric recomputation
```

The server sends one generated token ID per event, including EOS. It truncates
recorded prompts to the requested exact input token budget. Input IDs, output IDs
and decoded output are retained. A separate 20 ms sampler observes server RSS.
See [design and metric definitions](docs/design.md) and [protocol](docs/protocol.md).

## Reproduction

The eight-cell matrix replays six prompts per cell and performs one excluded
warmup before each cell. Generation is greedy with a 24-token maximum. The
reference run used the shared project-series exclusive benchmark lock:

```sh
python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py \
  .venv/bin/python -m serving_bench run --output results/reference
```

That coordinator script is external to this standalone repository. On another
machine, run the standard command above with no other heavy jobs. Do not bypass
the shared lock when reproducing inside the original project series.

[Raw report](results/reference/report.json) records hardware, Python and library
versions, source commit, model revision and SHA-256 file hashes, complete requests,
per-token arrival offsets, completion information, errors and RSS samples.
[requirements-local.lock](requirements-local.lock) records the local environment,
including development tools. It is not a universal wheel lock. Model weights are
not redistributed. See [model notices](docs/model-notices.md).

## Results

Results and interpretation are in [the measurement report](docs/results.md).
The raw evidence comes only from the machine identified in that report.

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m coverage run -m unittest discover -v
python -m coverage report --fail-under=90
python -m serving_bench verify results/reference/report.json
```

CI runs these checks on Linux, macOS and Windows using Python 3.11.9. Unit tests
exercise real HTTP sockets with a deterministic test adapter, malformed protocol
messages, token accounting, failure propagation, metric arithmetic, report
mutations, model API contracts and process cleanup. The test adapter is used only
in tests. CI does not download weights or repeat neural performance measurements.
The published benchmark separately exercises real neural inference.

## Limitations

- Six requests per cell are descriptive observations. Tail percentiles are not
  stable service-level estimates. No confidence interval or general speedup is claimed.
- A single model, host, precision and fixed cell order cannot characterize other
  hardware, larger models, batching engines or production traffic. No answer-quality
  evaluation is performed. Prompts are synthetic original English notes.
- Concurrency uses closed-loop workers with no arrival-rate schedule. TTFT includes
  queueing. No throughput saturation curve or continuous batching is measured.
- Client and server share the CPU. RSS is sampled process memory, not peak allocated
  tensor memory or system-wide usage. Short peaks may be missed.
- No calibrated energy sensor was available; energy is null. Cost is an illustrative
  full-machine allocation at a supplied USD/hour rate. Electricity, acquisition,
  utilization, idle time, downloads, warmup and labor are not measured expenses.
- The server is a trusted local benchmark fixture with no authentication, TLS or
  production hardening. Do not expose it to a network. Socket timeouts are inactivity
  limits, not strict end-to-end deadlines. It does not implement an OpenAI API.
- Verification checks internal consistency, not authenticity, statistical validity,
  or correctness against an independent implementation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Preserve raw measurements and add regression
tests for protocol or metric changes. Report issues through the public repository.

## License

Code, original prompts, documentation and measurements use [MIT](LICENSE).
SmolLM2 weights retain [Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
Other dependencies retain their respective upstream licenses.
