"""Closed-loop HTTP client. Each streamed event must contain exactly one token."""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from .metrics import number, request_metrics
from .server import validate_request


def request(url: str, body: dict, timeout: float = 120) -> dict:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("Only loopback HTTP endpoints are supported")
    validate_request(body)
    if number(timeout) == 0:
        raise ValueError("Timeout must be positive")
    row: dict = {
        "request": body,
        "token_ids": [],
        "arrival_s": [],
        "completion": None,
        "error": None,
    }
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("Expected SSE response")
            for line in response:
                if not line.strip():
                    continue
                if not line.startswith(b"data: "):
                    raise ValueError("Malformed SSE record")
                event = json.loads(line[6:])
                if event["type"] == "token":
                    token = event["id"]
                    if type(token) is not int or token < 0:
                        raise ValueError("Invalid token ID")
                    if len(row["token_ids"]) >= body["max_new_tokens"]:
                        raise ValueError("Output exceeds token budget")
                    row["arrival_s"].append(time.perf_counter() - start)
                    row["token_ids"].append(token)
                elif event["type"] == "done":
                    if event["input_tokens"] != body["input_tokens"]:
                        raise ValueError("Input token count mismatch")
                    row["completion"] = event
                    break
                else:
                    raise ValueError(event.get("message", "Unexpected event"))
            if row["completion"] is None:
                raise ValueError("Stream ended without completion")
        row["elapsed_s"] = time.perf_counter() - start
        request_metrics(row)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_s"] = time.perf_counter() - start
    return row


def run_requests(url: str, requests: list[dict], concurrency: int) -> tuple[list[dict], float]:
    if type(concurrency) is not int or not 1 <= concurrency <= 16 or not requests:
        raise ValueError("Require requests and concurrency in [1, 16]")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda body: request(url, body), requests))
    return rows, time.perf_counter() - start
