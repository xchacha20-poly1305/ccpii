# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers", "torch"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""A ccpii detection endpoint backed by any Hugging Face token-classification model.

    uv run --script hf_ner_server.py --model dslim/distilbert-NER --port 8765

then set the ccpii `endpoint` option to http://127.0.0.1:8765/detect.
The model's entity groups (PER, ORG, ...) become placeholder kinds as they are.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from transformers import pipeline


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="a token-classification model id or path")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labels", help="comma-separated entity groups to keep; all when left out")
    args = parser.parse_args()

    # "first" labels whole words: "simple" can split one into its sub-word pieces.
    tagger = pipeline("token-classification", model=args.model, aggregation_strategy="first")
    kept = set(args.labels.split(",")) if args.labels else None
    lock = Lock()

    def detect(text: str) -> list[dict[str, object]]:
        if not text.strip():
            return []
        with lock:
            entities = tagger(text)
        return [
            {
                "label": entity["entity_group"],
                # The pipeline counts Python characters; ccpii wants UTF-16 units.
                "start": utf16_length(text[: entity["start"]]),
                "end": utf16_length(text[: entity["end"]]),
            }
            for entity in entities
            if kept is None or entity["entity_group"] in kept
        ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
                status, payload = 200, {"spans": [detect(text) for text in texts]}
            except Exception as error:  # noqa: BLE001 - ccpii withholds on any non-2xx
                status, payload = 500, {"error": f"{type(error).__name__}: {error}"}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            pass

    # Loopback only: the texts are what ccpii is there to protect.
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
