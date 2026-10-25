# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "presidio-analyzer",
#   # 3.0.14 breaks loading every spaCy model: OverflowError in Tokenizer._flush_specials.
#   "preshed<3.0.14",
#   "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""A ccpii detection endpoint backed by Microsoft Presidio.

    uv run --script presidio_server.py --port 8765

then set the ccpii `endpoint` option to http://127.0.0.1:8765/detect.
Presidio's entity types (EMAIL_ADDRESS, PHONE_NUMBER, PERSON, ...) become
placeholder kinds as they are. It runs spaCy's small English model; for
another language or model, change NLP_CONFIG.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

NLP_CONFIG = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threshold", type=float, default=0.5, help="the lowest score kept")
    args = parser.parse_args()

    analyzer = AnalyzerEngine(nlp_engine=NlpEngineProvider(nlp_configuration=NLP_CONFIG).create_engine())
    lock = Lock()

    def detect(text: str) -> list[dict[str, object]]:
        if not text.strip():
            return []
        with lock:
            results = analyzer.analyze(text=text, language="en", score_threshold=args.threshold)
        return [
            {
                "label": result.entity_type,
                # Presidio counts Python characters; ccpii wants UTF-16 units.
                "start": utf16_length(text[: result.start]),
                "end": utf16_length(text[: result.end]),
            }
            for result in results
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
