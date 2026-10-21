# Reference protocol

Only loopback HTTP on 127.0.0.1 is accepted by the client. Proxy environment
variables are ignored. The server binds an ephemeral loopback port and atomically
writes readiness JSON for its parent process. GET /health responds after loading.

POST /generate accepts exactly three JSON keys: prompt (1 to 32000 characters),
input_tokens (integer 1 to 2048), max_new_tokens (integer 1 to 128). Booleans are
not integers in this contract. The HTTP body is limited to 128 KiB. The tokenizer
must yield at least input_tokens tokens; the prefix of that length is used.
Requests share one model and a generation lock. No fairness guarantee is made.

The SSE dialect uses a single JSON data line per event, followed by a blank line:

```text
data: {"type": "token", "id": 42}

data: {"type": "done", "input_tokens": 64, "output_tokens": 1, "input_ids": [], "text": "example", "finish_reason": "eos"}
```

The example omits the 64 actual input IDs for brevity. Actual completions retain
all IDs. A done event terminates the client request and must match observed token
count and requested input count. An error event, malformed event, early EOF or
HTTP error records failure; partial tokens remain visible but do not contribute
to successful throughput or latency percentiles. A token ID of zero is valid.
Server writes are flushed per token. Socket buffering and OS scheduling can still
coalesce client observations. No equivalence with external SSE APIs is claimed.
