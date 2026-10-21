"""Command-line entry point."""

import argparse
import json
import sys
from pathlib import Path

from .experiment import download, run, verify
from .server import LocalModel, make_server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("download")
    get.add_argument("--model", type=Path, default=Path(".cache/model"))
    serve = sub.add_parser("serve")
    serve.add_argument("--model", type=Path, default=Path(".cache/model"))
    serve.add_argument("--threads", type=int, choices=(1, 2), default=1)
    serve.add_argument("--ready", type=Path, required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--model", type=Path, default=Path(".cache/model"))
    bench.add_argument("--corpus", type=Path, default=Path("data/prompts.json"))
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--hourly-usd", type=float, default=0.10)
    check = sub.add_parser("verify")
    check.add_argument("report", type=Path)
    check.add_argument("--corpus", type=Path, default=Path("data/prompts.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download(args.model)
        elif args.command == "serve":
            server = make_server(LocalModel(args.model, args.threads))
            temporary = args.ready.with_suffix(".tmp")
            temporary.write_text(json.dumps({"port": server.server_port}))
            temporary.replace(args.ready)
            try:
                server.serve_forever()
            finally:
                server.server_close()
        elif args.command == "run":
            run(args.model, args.corpus, args.output, args.hourly_usd)
        else:
            verify(json.loads(args.report.read_text()), args.corpus)
            print("Report verified")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
