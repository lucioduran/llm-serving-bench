import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from serving_bench.__main__ import main
from serving_bench.experiment import download, measure, run
from serving_bench.server import LocalModel


class RuntimeTests(unittest.TestCase):
    def test_local_model_streaming_contract(self):
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        tokenizer = Mock(eos_token_id=2)
        tokenizer.encode.return_value = [4, 5, 6]
        tokenizer.decode.return_value = "answer"
        model = Mock()
        output = Mock()
        output.__getitem__ = Mock(return_value=Mock(tolist=lambda: [7, 2]))

        def generate(**kwargs):
            kwargs["streamer"].put(Mock())
            kwargs["streamer"].put(Mock(tolist=lambda: [7]))
            kwargs["streamer"].put(Mock(tolist=lambda: [2]))
            return output

        model.generate.side_effect = generate
        transformers = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers.AutoModelForCausalLM.from_pretrained.return_value.eval.return_value = model
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            backend = LocalModel(Path("model"), 2)
            events = []
            result = backend.generate(
                {"prompt": "text", "input_tokens": 2, "max_new_tokens": 2}, events.append
            )
            self.assertEqual(result["input_ids"], [4, 5])
            self.assertEqual(result["output_tokens"], 2)
            self.assertEqual(result["finish_reason"], "eos")
            self.assertEqual(events, [{"type": "token", "id": 7}, {"type": "token", "id": 2}])
            torch.set_num_threads.assert_called_once_with(2)
            torch.set_num_interop_threads.assert_called_once_with(1)
            with self.assertRaises(ValueError):
                backend.generate({"prompt": "text", "input_tokens": 4}, events.append)

    def test_download_pin(self):
        hub = Mock()
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            download(Path("model"))
        kwargs = hub.snapshot_download.call_args.kwargs
        self.assertEqual(len(kwargs["revision"]), 40)
        self.assertFalse(kwargs["token"])
        self.assertIn("model.safetensors", kwargs["allow_patterns"])

    def test_memory_sampling(self):
        psutil = Mock()
        psutil.Process.return_value.memory_info.return_value.rss = 123
        with (
            patch.dict("sys.modules", {"psutil": psutil}),
            patch("serving_bench.experiment.run_requests", return_value=([{"result": 1}], 0.1)),
        ):
            result = measure("url", [{}], 1, 1234)
        self.assertEqual(result["rss_samples"][0]["rss_bytes"], 123)
        self.assertEqual(result["wall_s"], 0.1)

    def test_experiment_orchestration_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"fixture")
            corpus = root / "corpus.json"
            corpus.write_text(json.dumps(["hello"] * 6))
            psutil = Mock()
            psutil.cpu_count.return_value = 2
            psutil.virtual_memory.return_value.total = 1024
            process = Mock(pid=1234)
            cell = {
                "requests": [
                    {
                        "elapsed_s": 0.1,
                        "arrival_s": [0.05],
                        "token_ids": [1],
                        "error": None,
                        "completion": {"output_tokens": 1},
                    }
                ],
                "wall_s": 0.2,
                "rss_samples": [{"rss_bytes": 123, "offset_s": 0}],
            }
            with (
                patch.dict("sys.modules", {"psutil": psutil}),
                patch("serving_bench.experiment.platform.platform", return_value="test"),
                patch("serving_bench.experiment.platform.processor", return_value="test"),
                patch("serving_bench.experiment.subprocess.check_output", return_value="test"),
                patch("serving_bench.experiment.importlib.metadata.version", return_value="test"),
                patch("serving_bench.experiment.subprocess.Popen", return_value=process),
                patch("serving_bench.experiment.wait_ready", return_value=1234),
                patch("serving_bench.experiment.request", return_value={"error": None}),
                patch("serving_bench.experiment.measure", side_effect=lambda *a: dict(cell)),
            ):
                result = run(model, corpus, root / "output", 0.1)
                self.assertEqual(len(result["cells"]), 8)
                self.assertEqual(process.terminate.call_count, 2)
                self.assertTrue((root / "output/report.json").is_file())
                with self.assertRaises(FileExistsError):
                    run(model, corpus, root / "output", 0.1)
                with patch("serving_bench.experiment.request", return_value={"error": "failed"}):
                    with self.assertRaises(RuntimeError):
                        run(model, corpus, root / "failed", 0.1)
                self.assertEqual(process.terminate.call_count, 3)
            corpus.write_text("[]")
            with self.assertRaises(ValueError):
                run(model, corpus, root / "bad", 0.1)
            corpus.write_text(json.dumps(["hello"] * 6))
            with self.assertRaises(ValueError):
                run(root / "missing", corpus, root / "bad", 0.1)

    def test_cli_dispatch(self):
        with patch("serving_bench.__main__.download") as get:
            self.assertEqual(main(["download"]), 0)
            get.assert_called_once()
        with patch("serving_bench.__main__.run") as bench:
            self.assertEqual(main(["run", "--output", "unused"]), 0)
            bench.assert_called_once()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("serving_bench.__main__.LocalModel"),
            patch("serving_bench.__main__.make_server") as factory,
        ):
            factory.return_value.server_port = 1234
            ready = Path(directory) / "ready.json"
            self.assertEqual(main(["serve", "--ready", str(ready)]), 0)
            self.assertEqual(json.loads(ready.read_text())["port"], 1234)
            factory.return_value.server_close.assert_called_once()
