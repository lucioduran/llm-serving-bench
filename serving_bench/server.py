"""Loopback-only single-slot reference server with token-level SSE."""

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def validate_request(body: object) -> dict:
    if not isinstance(body, dict) or set(body) != {"prompt", "input_tokens", "max_new_tokens"}:
        raise ValueError("Expected prompt, input_tokens and max_new_tokens")
    if not isinstance(body["prompt"], str) or not 1 <= len(body["prompt"]) <= 32000:
        raise ValueError("Invalid prompt")
    for name, limit in (("input_tokens", 2048), ("max_new_tokens", 128)):
        if type(body[name]) is not int or not 1 <= body[name] <= limit:
            raise ValueError(f"Invalid {name}")
    return body


class TokenStream:
    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit
        self.prompt = True
        self.count = 0

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        ids = value.tolist()
        if len(ids) != 1 or type(ids[0]) is not int:
            raise ValueError("Only single-sequence generation is supported")
        self.count += 1
        self.emit({"type": "token", "id": ids[0]})

    def end(self):
        pass


class LocalModel:
    def __init__(self, path: Path, threads: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()

    def generate(self, body: dict, emit: Callable[[dict], None]) -> dict:
        ids = self.tokenizer.encode(body["prompt"], add_special_tokens=False)
        count = body["input_tokens"]
        if len(ids) < count:
            raise ValueError("Prompt shorter than requested input token budget")
        ids = ids[:count]
        stream = TokenStream(emit)
        tensor = self.torch.tensor([ids])
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=tensor,
                attention_mask=self.torch.ones_like(tensor),
                max_new_tokens=body["max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=stream,
            )
        generated = output[0, count:].tolist()
        return {
            "type": "done",
            "input_tokens": count,
            "output_tokens": stream.count,
            "input_ids": ids,
            "text": self.tokenizer.decode(generated),
            "finish_reason": "eos" if generated[-1] == self.tokenizer.eos_token_id else "length",
        }


def make_server(model, port: int = 0) -> ThreadingHTTPServer:
    gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ready")

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131072:
                    raise ValueError("Invalid body size")
                self.connection.settimeout(120)
                body = validate_request(json.loads(self.rfile.read(length)))
            except (ValueError, OSError):
                self.send_error(400, "Invalid request")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(event):
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()

            try:
                with gate:
                    emit(model.generate(body, emit))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            except Exception as exc:
                emit({"type": "error", "message": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
