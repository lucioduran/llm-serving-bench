# Observed reference results

Recorded on 2026-10-21 UTC, using an Intel Xeon W-3223 at 3.50 GHz,
8 physical / 16 logical CPUs, 48 GiB RAM, macOS 26.7.1 x86_64.
Inference used only 1 or 2 CPU intra-op threads, inter-op threads 1, FP32,
PyTorch 2.2.2 and Transformers 4.46.3 under Python 3.11.13.

All 48 measured requests succeeded and generated 24 tokens each (1,152 total).
Eight additional warmups are retained but excluded from summaries. No GPU was used.

| Threads | Input tokens | Concurrency | TTFT p50 (s) | TPOT p50 (ms) | Latency p95 (s) | Output tokens/s | Peak RSS (MiB) | Assumed USD/1M tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 1 | 0.196 | 62.92 | 1.704 | 14.49 | 876.9 | 1.92 |
| 1 | 64 | 2 | 1.871 | 63.74 | 3.355 | 14.35 | 892.3 | 1.94 |
| 1 | 256 | 1 | 0.654 | 72.02 | 2.415 | 10.28 | 1009.3 | 2.70 |
| 1 | 256 | 2 | 3.002 | 72.84 | 4.691 | 10.24 | 1023.5 | 2.71 |
| 2 | 64 | 1 | 0.128 | 42.89 | 1.143 | 21.50 | 838.7 | 1.29 |
| 2 | 64 | 2 | 1.256 | 43.24 | 2.460 | 20.80 | 858.1 | 1.34 |
| 2 | 256 | 1 | 0.414 | 48.94 | 1.601 | 15.51 | 959.6 | 1.79 |
| 2 | 256 | 2 | 2.041 | 50.61 | 3.234 | 14.93 | 997.6 | 1.86 |

Two threads produced higher throughput than one in these cells. Concurrency two
mostly adds queueing: one slot still generates a single sequence at a time.
The longer context increases prefill time. The sample is too small to establish
a stable ranking, tail-latency guarantee or general hardware recommendation.

Cost uses an **illustrative USD 0.10 per full machine-hour**, allocated only over
each measured cell. It is not a measured bill or electricity charge. Paid API
and rented-compute charges were zero. Energy is unavailable and remains null;
there was no calibrated sensor. No model quality or security evaluation is claimed.

The [raw report](../results/reference/report.json) retains p50/p95/p99 values,
individual arrivals, all requests and input IDs, RSS samples and full provenance.
Measured source revision: `1f78bb2b7926840bbec088c289d56285daac807c` with a clean working tree.
Later commits add tests, documentation and evidence without changing the measured
server, client or experiment implementation. Fixed order, shared host activity,
client CPU usage and only six samples per cell limit the conclusions.

Verification: 20 local unit/contract tests, 98% combined statement/branch coverage,
Ruff lint and formatting, mypy, and offline recomputation of all eight summaries.
The release is gated on CI passing for its exact final commit.
