# Measurement design

A loopback HTTP server holds one FP32 SmolLM2 model. A generation lock admits one
request at a time. Compare PyTorch intra-op thread counts 1 and 2 with inter-op
threads fixed to 1. No batching, GPU, prefix reuse, quantization or hosted calls.
The client applies closed-loop concurrency 1 and 2. Each cell replays six recorded
prompts at exact input budgets 64 and 256 tokens, generating up to 24 tokens.
Greedy decoding can stop early at EOS. Warm up once per context before each cell.

The server streams one token ID per SSE event, including special tokens. The
client records monotonic arrival offsets starting before HTTP connection setup.
TTFT includes network setup, tokenization, queueing and prefill. TPOT is
(last arrival - first arrival)/(token count - 1), null for fewer than two tokens.
End-to-end latency ends at the validated completion event. Linear interpolation
computes p50/p95/p99 on successful requests; errors remain in the denominator.
No distributional confidence is implied by six samples. A process RSS sampler
runs every 20 ms during each measured cell; peaks can miss short-lived allocations.

Server and client share the machine. Fixed cell order and OS background work can
confound comparisons. The external coordinator lock serializes heavy project
experiments but does not isolate the OS. Energy is null without a real sensor.
A user-supplied assumed machine USD/hour yields allocation cost, not an invoice
or electricity measurement. Default illustrative rate is USD 0.10/hour, at full
allocation during each cell. Downloads, warmup, idle time and labor are excluded.

Raw reports retain requests, token IDs, arrival offsets, failures, RSS samples,
wall durations, model revision and hashes, runtime packages, hardware and source
revision. Offline verification recomputes summaries and checks timing invariants.
It establishes internal consistency, not provenance or independent review.
