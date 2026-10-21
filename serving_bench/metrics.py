"""Recomputable client-side measurements."""

import math


def number(value: object, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError("Number outside permitted range")
    return float(value)


def percentile(values: list[float], q: float) -> float | None:
    if not 0 <= q <= 1:
        raise ValueError("Quantile outside [0, 1]")
    if not values:
        return None
    ordered = sorted(number(v) for v in values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def request_metrics(row: dict) -> dict:
    elapsed = number(row["elapsed_s"])
    arrivals = [number(v) for v in row["arrival_s"]]
    tokens = row["token_ids"]
    if len(tokens) != len(arrivals) or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Token and arrival mismatch")
    if arrivals != sorted(arrivals) or (arrivals and arrivals[-1] > elapsed):
        raise ValueError("Invalid arrival ordering")
    if row["error"] is not None:
        return {"ttft_s": None, "tpot_s": None, "latency_s": None}
    if not arrivals or row["completion"]["output_tokens"] != len(tokens):
        raise ValueError("Incomplete successful response")
    return {
        "ttft_s": arrivals[0],
        "tpot_s": (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) if len(arrivals) > 1 else None,
        "latency_s": elapsed,
    }


def summarize(cell: dict, hourly_usd: float) -> dict:
    rate = number(hourly_usd)
    wall = number(cell["wall_s"])
    if wall == 0 or not cell["requests"]:
        raise ValueError("Empty measurement")
    metrics = [request_metrics(row) for row in cell["requests"]]
    successes = sum(row["error"] is None for row in cell["requests"])
    tokens = sum(len(row["token_ids"]) for row in cell["requests"] if row["error"] is None)
    result = {
        "requests": len(metrics),
        "successes": successes,
        "errors": len(metrics) - successes,
        "output_tokens": tokens,
        "tokens_per_s": tokens / wall,
        "requests_per_s": successes / wall,
        "rss_peak_bytes": max((s["rss_bytes"] for s in cell["rss_samples"]), default=None),
        "energy_joules": None,
        "assumed_allocated_usd": wall / 3600 * rate,
        "assumed_usd_per_million_tokens": wall / 3600 * rate / tokens * 1e6 if tokens else None,
    }
    for field in ("ttft_s", "tpot_s", "latency_s"):
        values = [m[field] for m in metrics if m[field] is not None]
        result[field] = {f"p{int(q * 100)}": percentile(values, q) for q in (0.5, 0.95, 0.99)}
    return result
