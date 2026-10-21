import copy
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from serving_bench.__main__ import main
from serving_bench.client import request, run_requests
from serving_bench.experiment import digest, verify, wait_ready
from serving_bench.metrics import number, percentile, request_metrics, summarize
from serving_bench.server import TokenStream, make_server, validate_request

BODY = {"prompt": "hello", "input_tokens": 1, "max_new_tokens": 3}


class FakeModel:
    def generate(self, body, emit):
        emit({"type": "token", "id": 10})
        emit({"type": "token", "id": 11})
        return {
            "type": "done",
            "input_tokens": body["input_tokens"],
            "input_ids": [1] * body["input_tokens"],
            "output_tokens": 2,
        }


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.server = make_server(self.model)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_stream_and_closed_loop(self):
        rows, wall = run_requests(self.url + "/generate", [BODY] * 4, 2)
        self.assertGreater(wall, 0)
        for row in rows:
            self.assertIsNone(row["error"])
            self.assertEqual(row["token_ids"], [10, 11])
            self.assertGreater(request_metrics(row)["ttft_s"], 0)
        with urllib.request.urlopen(self.url + "/health") as response:
            self.assertEqual(response.read(), b"ready")

    def test_bad_routes_and_body(self):
        for path, data, code in [
            ("/missing", None, 404),
            ("/missing", b"{}", 404),
            ("/generate", b"{", 400),
            ("/generate", b"{}", 400),
            ("/generate", b"", 400),
        ]:
            req = urllib.request.Request(self.url + path, data=data)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code, code)

    def test_model_failure_and_incomplete_stream(self):
        self.model.generate = Mock(side_effect=ValueError("generation failed"))
        row = request(self.url + "/generate", BODY)
        self.assertIn("generation failed", row["error"])
        self.model.generate = Mock(
            return_value={"type": "done", "input_tokens": 1, "output_tokens": 0}
        )
        row = request(self.url + "/generate", BODY)
        self.assertIn("Incomplete", row["error"])

    def test_mismatched_counts(self):
        self.model.generate = Mock(return_value={"type": "done", "input_tokens": 2})
        self.assertIn("mismatch", request(self.url + "/generate", BODY)["error"])


class ProtocolTests(unittest.TestCase):
    def response(self, payload, content_type="text/event-stream"):
        response = io.BytesIO(payload)
        response.headers = Mock()
        response.headers.get_content_type.return_value = content_type
        opener = Mock()
        opener.open.return_value = response
        return patch("urllib.request.build_opener", return_value=opener)

    def test_rejects_bad_streams(self):
        for data in [
            b"",
            b"broken\n",
            b"data: {}\n",
            b"data: null\n",
            b'data: {"type":"token","id":true}\n',
            b'data: {"type":"token","id":-1}\n',
            b'data: {"type":"unknown"}\n',
            b'data: {"type":"token","id":1}\n' * 4,
        ]:
            with self.subTest(data=data), self.response(data):
                self.assertIsNotNone(request("http://127.0.0.1/generate", BODY)["error"])
        with self.response(b"", "text/html"):
            self.assertIn("SSE", request("http://127.0.0.1/generate", BODY)["error"])

    def test_invalid_client_arguments(self):
        for url in ["https://127.0.0.1/", "http://example.com/", "file:///tmp/data"]:
            with self.assertRaises(ValueError):
                request(url, BODY)
        with self.assertRaises(ValueError):
            request("http://127.0.0.1/", BODY, timeout=0)
        for concurrency in [0, 17, True]:
            with self.assertRaises(ValueError):
                run_requests("", [BODY], concurrency)
        with self.assertRaises(ValueError):
            run_requests("", [], 1)

    def test_streamer_skips_prompt_not_empty_text_tokens(self):
        events = []
        stream = TokenStream(events.append)
        stream.put(Mock())
        stream.put(Mock(tolist=lambda: [0]))
        stream.end()
        self.assertEqual(events, [{"type": "token", "id": 0}])
        with self.assertRaises(ValueError):
            stream.put(Mock(tolist=lambda: [1, 2]))

    def test_request_validation(self):
        self.assertEqual(validate_request(BODY), BODY)
        for body in [
            None,
            [],
            {},
            {**BODY, "extra": 1},
            {**BODY, "prompt": ""},
            {**BODY, "prompt": 3},
            {**BODY, "input_tokens": True},
            {**BODY, "input_tokens": 2049},
            {**BODY, "max_new_tokens": 0},
        ]:
            with self.assertRaises(ValueError):
                validate_request(body)


class MetricTests(unittest.TestCase):
    def row(self):
        return {
            "elapsed_s": 4.0,
            "arrival_s": [1.0, 2.0, 3.0],
            "token_ids": [1, 2, 3],
            "error": None,
            "completion": {"output_tokens": 3},
        }

    def test_known_metrics(self):
        self.assertEqual(request_metrics(self.row()), {"ttft_s": 1, "tpot_s": 1, "latency_s": 4})
        self.assertEqual(percentile([1, 2, 3], 0.95), 2.9)
        self.assertIsNone(percentile([], 0.5))
        row = {**self.row(), "arrival_s": [1], "token_ids": [1], "completion": {"output_tokens": 1}}
        self.assertIsNone(request_metrics(row)["tpot_s"])
        cell = {
            "wall_s": 5,
            "requests": [self.row(), {**self.row(), "error": "failed"}],
            "rss_samples": [{"rss_bytes": 10}, {"rss_bytes": 20}],
        }
        summary = summarize(cell, 0.36)
        self.assertEqual(summary["tokens_per_s"], 0.6)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["rss_peak_bytes"], 20)
        self.assertEqual(summary["assumed_allocated_usd"], 0.0005)
        self.assertIsNone(summary["energy_joules"])
        cell["requests"] = [{**self.row(), "error": "failed"}]
        self.assertIsNone(summarize(cell, 0)["assumed_usd_per_million_tokens"])

    def test_invalid_measurements(self):
        for value in [True, "1", -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                number(value)
        with self.assertRaises(ValueError):
            percentile([1], 2)
        for change in [
            {"arrival_s": [2, 1, 3]},
            {"arrival_s": [1, 2, 5]},
            {"token_ids": [1]},
            {"token_ids": [1, 2, -1]},
            {"completion": {"output_tokens": 4}},
        ]:
            with self.assertRaises(ValueError):
                request_metrics({**self.row(), **change})
        with self.assertRaises(ValueError):
            summarize({"wall_s": 0, "requests": []}, 0)


class EvidenceTests(unittest.TestCase):
    def fixture(self, corpus):
        prompts = ["hello"] * 6
        corpus.write_text(json.dumps(prompts))
        report = {
            "schema_version": 1,
            "corpus_sha256": digest(corpus),
            "assumptions": {"machine_usd_per_hour": 0},
            "cells": [],
        }
        for threads in (1, 2):
            for context in (64, 256):
                for concurrency in (1, 2):
                    row = {
                        "elapsed_s": 1.0,
                        "arrival_s": [0.5],
                        "token_ids": [1],
                        "error": None,
                        "completion": {
                            "input_tokens": context,
                            "input_ids": [1] * context,
                            "output_tokens": 1,
                        },
                        "request": {
                            "prompt": "hello",
                            "input_tokens": context,
                            "max_new_tokens": 24,
                        },
                    }
                    cell = {
                        "threads": threads,
                        "context_tokens": context,
                        "concurrency": concurrency,
                        "requests": [row] * 6,
                        "wall_s": 6.0,
                        "rss_samples": [{"offset_s": 0, "rss_bytes": 1}],
                    }
                    cell["summary"] = summarize(cell, 0)
                    report["cells"].append(cell)
        return report

    def test_replay_and_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            report = self.fixture(corpus)
            verify(report, corpus)
            mutations = [
                lambda r: r.update(schema_version=2),
                lambda r: r.update(corpus_sha256="bad"),
                lambda r: r["cells"].pop(),
                lambda r: r["cells"][0]["requests"].pop(),
                lambda r: r["cells"][0].update(rss_samples=[]),
                lambda r: r["cells"][0]["summary"].update(errors=9),
                lambda r: r["cells"][0]["requests"][0].update(elapsed_s=7),
                lambda r: r["cells"][0]["requests"][0]["request"].update(prompt="bad"),
                lambda r: r["cells"][0]["requests"][0]["completion"].update(input_ids=[]),
                lambda r: r["cells"][0]["rss_samples"][0].update(rss_bytes=-1),
            ]
            for mutate in mutations:
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    verify(changed, corpus)
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(main(["verify", str(path), "--corpus", str(corpus)]), 0)
            path.write_text("{}")
            self.assertEqual(main(["verify", str(path), "--corpus", str(corpus)]), 2)

    def test_startup_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.json"
            with self.assertRaises(RuntimeError):
                wait_ready(Mock(poll=lambda: 1), path)
            with self.assertRaises(TimeoutError):
                wait_ready(Mock(poll=lambda: None), path, timeout=0)
            path.write_text('{"port": 1234}')
            self.assertEqual(wait_ready(Mock(poll=lambda: None), path), 1234)
