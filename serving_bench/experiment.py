"""Download pinned artifacts and measure a bounded local experiment."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from .client import request, run_requests
from .metrics import number, summarize
from .server import MODEL_ID

REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=path,
        token=False,
        allow_patterns=[
            "*.json",
            "model.safetensors",
            "LICENSE*",
            "README.md",
            "merges.txt",
            "vocab.json",
        ],
    )


def wait_ready(process, path: Path, timeout: float = 120) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before readiness; inspect server log")
        if path.exists():
            return json.loads(path.read_text())["port"]
        time.sleep(0.05)
    raise TimeoutError("Server startup timed out")


def measure(url: str, bodies: list[dict], concurrency: int, pid: int) -> dict:
    import psutil

    process = psutil.Process(pid)
    stop = threading.Event()
    samples: list[dict] = []
    start = time.perf_counter()

    def sample():
        while not stop.is_set():
            samples.append(
                {"offset_s": time.perf_counter() - start, "rss_bytes": process.memory_info().rss}
            )
            stop.wait(0.02)

    worker = threading.Thread(target=sample)
    worker.start()
    try:
        rows, wall = run_requests(url, bodies, concurrency)
    finally:
        stop.set()
        worker.join()
    if not samples:
        raise RuntimeError("No memory samples collected")
    return {"requests": rows, "wall_s": wall, "rss_samples": samples}


def run(model: Path, corpus: Path, output: Path, hourly_usd: float) -> dict:
    number(hourly_usd)
    prompts = json.loads(corpus.read_text())
    if (
        not isinstance(prompts, list)
        or len(prompts) != 6
        or not all(isinstance(p, str) and p for p in prompts)
    ):
        raise ValueError("Corpus must contain six nonempty prompts")
    if not (model / "model.safetensors").is_file():
        raise ValueError("Download the model first")
    output.mkdir(parents=True, exist_ok=False)
    import psutil

    report: dict = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if sys.platform == "darwin"
            else platform.processor(),
            "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "device": "cpu",
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "numpy",
                    "psutil",
                    "tokenizers",
                    "safetensors",
                )
            },
        },
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "dtype": "float32",
            "files": {p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()},
        },
        "corpus_sha256": digest(corpus),
        "assumptions": {
            "machine_usd_per_hour": hourly_usd,
            "energy_sensor": None,
            "energy_reason": "No calibrated energy sensor available",
            "cost_basis": "Full machine allocation during measured cells only",
        },
        "cells": [],
    }
    for threads in (1, 2):
        ready = output / f"ready-{threads}.json"
        env = {
            **os.environ,
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
        }
        with (output / f"server-{threads}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "serving_bench",
                    "serve",
                    "--model",
                    str(model.resolve()),
                    "--threads",
                    str(threads),
                    "--ready",
                    str(ready.resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            try:
                port = wait_ready(process, ready)
                url = f"http://127.0.0.1:{port}/generate"
                for context in (64, 256):
                    bodies = [
                        {"prompt": p, "input_tokens": context, "max_new_tokens": 24}
                        for p in prompts
                    ]
                    for concurrency in (1, 2):
                        warmup = request(url, bodies[0])
                        if warmup["error"]:
                            raise RuntimeError(f"Warmup failed: {warmup['error']}")
                        cell = measure(url, bodies, concurrency, process.pid)
                        cell.update(
                            threads=threads,
                            context_tokens=context,
                            concurrency=concurrency,
                            warmup=warmup,
                        )
                        cell["summary"] = summarize(cell, hourly_usd)
                        report["cells"].append(cell)
                        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            json.dumps(
                                {
                                    "threads": threads,
                                    "context": context,
                                    "concurrency": concurrency,
                                    "summary": cell["summary"],
                                }
                            ),
                            flush=True,
                        )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                ready.unlink(missing_ok=True)
    return report


def verify(report: dict, corpus: Path) -> None:
    if report["schema_version"] != 1 or report["corpus_sha256"] != digest(corpus):
        raise ValueError("Schema or corpus mismatch")
    prompts = json.loads(corpus.read_text())
    expected = [(t, c, n) for t in (1, 2) for c in (64, 256) for n in (1, 2)]
    actual = [(c["threads"], c["context_tokens"], c["concurrency"]) for c in report["cells"]]
    if actual != expected:
        raise ValueError("Incomplete or reordered experiment matrix")
    for cell in report["cells"]:
        if len(cell["requests"]) != len(prompts) or not cell["rss_samples"]:
            raise ValueError("Incomplete cell")
        previous = -1.0
        for sample in cell["rss_samples"]:
            offset = number(sample["offset_s"])
            if (
                offset < previous
                or type(sample["rss_bytes"]) is not int
                or sample["rss_bytes"] <= 0
            ):
                raise ValueError("Invalid memory sample")
            previous = offset
        for prompt, row in zip(prompts, cell["requests"], strict=True):
            if row["request"] != {
                "prompt": prompt,
                "input_tokens": cell["context_tokens"],
                "max_new_tokens": 24,
            }:
                raise ValueError("Recorded request mismatch")
            if row["elapsed_s"] > cell["wall_s"] or len(row["token_ids"]) > 24:
                raise ValueError("Invalid request bounds")
            if row["error"] is None:
                completion = row["completion"]
                if (
                    completion["input_tokens"] != cell["context_tokens"]
                    or len(completion["input_ids"]) != cell["context_tokens"]
                ):
                    raise ValueError("Context mismatch")
        if summarize(cell, report["assumptions"]["machine_usd_per_hour"]) != cell["summary"]:
            raise ValueError("Summary differs from raw measurements")
