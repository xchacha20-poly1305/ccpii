# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "onnxruntime",
#   "numpy",
#   "huggingface_hub",
#   "opf @ git+https://github.com/openai/privacy-filter@f7f00ca7fb869683eb732c010299d901457f19c3",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Span detection service for the ccpii mod.

The forward pass runs the model's int8 ONNX export on onnxruntime; the rest
is opf's own pipeline (tokenizing, windows, constrained Viterbi decoding,
character offsets). opf's PyTorch forward pass is a reference implementation
on CPU, around fifty times slower, so only its pipeline is used, through its
internal modules: hence the pinned opf commit.

Two modes:

`--download` installs the model. It downloads into a staging directory and
renames it into place only once complete, under a lock, so the model
directory existing always means the model is whole, and two installs never
write the same files. An interrupted download resumes.

`--serve` serves `POST /detect` over a Unix socket in a private directory
of its own under /tmp, named on its `listening` line. The body is
`{"texts": [...]}`; the answer is `{"spans": [[{"label", "start", "end"}]]}`,
one list per text, offsets in UTF-16 code units so the JavaScript side can
slice directly. It never downloads: without an installed model it reports
`missing` and exits. The socket is bound before the model loads and
requests wait for the load. On SIGTERM or SIGINT it removes the socket and
its directory; after a SIGKILL whoever starts the next one removes them.

Status lines go to stdout, one word first: `downloading`, `installed`,
`missing`, `listening <socket path>`, `loading`, `ready`, `failed <reason>`,
and while serving `gpu starting` (a GPU model is built, its worker starts),
`device cpu` or `device gpu` (which one answers now) and `fallback <reason>`
(the GPU failed and the CPU took over; see GpuForward).

With an OpenVINO model for the GPU at `GPU_MODEL` (GPU.md), the forward pass
runs on the GPU in a worker process, `gpu_worker.py`, and on the CPU only
while the GPU is not answering.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = "openai/privacy-filter"
MODEL_DIR = Path.home() / ".cache" / "ccpii" / "privacy-filter"
STAGING = MODEL_DIR.with_name(MODEL_DIR.name + ".partial")
LOCK = MODEL_DIR.with_name(MODEL_DIR.name + ".lock")
# Repository path → name in MODEL_DIR. The checkpoint files opf's pipeline
# reads (its config and Viterbi calibration) and the ONNX export.
FILES = {
    "original/config.json": "config.json",
    "original/viterbi_calibration.json": "viterbi_calibration.json",
    # The graph names its weights file, so both keep their own names.
    "onnx/model_quantized.onnx": "model_quantized.onnx",
    "onnx/model_quantized.onnx_data": "model_quantized.onnx_data",
}
ONNX_MODEL = MODEL_DIR / "model_quantized.onnx"
# An OpenVINO model of your own making (GPU.md), run by `gpu_worker.py`.
GPU_MODEL = MODEL_DIR / "openvino-int8" / "model.xml"
GPU_WORKER = Path(__file__).with_name("gpu_worker.py")
# tiktoken caches its encoding under $TMPDIR by default, which the system
# may clear; kept beside the model, serving never needs the network.
os.environ["TIKTOKEN_CACHE_DIR"] = str(MODEL_DIR / "tiktoken")


def announce(line: str) -> None:
    print(line, flush=True)


def is_installed() -> bool:
    return all((MODEL_DIR / name).is_file() for name in FILES.values())


def download() -> int:
    if is_installed():
        announce("installed")
        return 0
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            announce("failed another install is running")
            return 1
        announce("downloading")
        # Plain HTTP rather than the Xet protocol, whose transfers were seen to
        # stall for good partway through; set it to 0 to try Xet anyway.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=REPO, local_dir=str(STAGING), allow_patterns=list(FILES))
            # The model directory is laid out in a sibling, then renamed whole.
            laid = STAGING / "model"
            laid.mkdir(exist_ok=True)
            for source, name in FILES.items():
                downloaded = STAGING / source
                if downloaded.is_file():
                    downloaded.rename(laid / name)
                if not (laid / name).is_file():
                    raise RuntimeError(f"the download has no {source}")
            os.environ["TIKTOKEN_CACHE_DIR"] = str(laid / "tiktoken")
            load_encoding(laid)
            laid.rename(MODEL_DIR)
            shutil.rmtree(STAGING)
        except Exception as error:  # noqa: BLE001 - reported to the mod
            announce(f"failed {type(error).__name__}: {error}")
            return 1
    announce("installed")
    return 0


def load_encoding(model_dir: Path):
    import tiktoken
    from opf._core.runtime import _load_checkpoint_config

    config = _load_checkpoint_config(str(model_dir))
    return config, tiktoken.get_encoding(str(config["encoding"]))


class OnnxForward:
    """The model's forward pass as opf's pipeline calls it: tokens in, logits out."""

    def __init__(self, path: Path) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Without the arena a 4096-token window leaves no buffers behind: 2.2 GB
        # resident in place of 5.5 GB, for about a quarter more time per call.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self._session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def __call__(self, tokens, attention_mask):
        import numpy
        import torch

        (logits,) = self._session.run(
            None,
            {
                "input_ids": tokens.numpy().astype(numpy.int64),
                "attention_mask": attention_mask.numpy().astype(numpy.int64),
            },
        )
        return torch.from_numpy(logits)


class GpuForward:
    """The forward pass on the GPU, in `gpu_worker.py`, under a watchdog.

    With system memory short, i915 has been seen to hang the worker's compute
    context: the call then never returns and the process ignores every signal
    but SIGKILL. So each call waits `ANSWER_SECONDS` at most; past that, or on
    an error, the worker is killed, the call returns None for the CPU to
    answer, and a new worker starts after a delay that doubles with each
    failure in a row, once enough memory is available. Calls before the
    worker is ready also return None.
    """

    # A 4096-token window takes under a second.
    ANSWER_SECONDS = 5.0
    # Compiling, from the model cache after the first time, and running each
    # input size once.
    START_SECONDS = 180.0
    FIRST_RETRY_SECONDS = 30.0
    LAST_RETRY_SECONDS = 600.0
    # A worker takes about 2 GB while it starts. Below this, starting one
    # deepens the shortage that most likely failed the last, so the retry
    # waits another round instead, not counted as a failure.
    RETRY_LEAST_AVAILABLE_MB = 4096

    def __init__(self, model: Path) -> None:
        self._model = model
        self._worker: subprocess.Popen | None = None
        self._ready = False
        self._failures = 0

    def start(self) -> None:
        """Starts a worker and waits for it in a thread of its own."""
        worker = subprocess.Popen(
            [shutil.which("uv") or "uv", "run", "--quiet", "--script", str(GPU_WORKER), str(self._model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # uv runs the worker as a child of its own and cannot pass SIGKILL
            # on, so the pair gets a process group, killed whole.
            start_new_session=True,
        )
        self._worker = worker
        threading.Thread(target=self._wait_ready, args=(worker,), daemon=True).start()

    def _wait_ready(self, worker: subprocess.Popen) -> None:
        try:
            ready = read_exactly(worker.stdout, 4, self.START_SECONDS) == struct.pack("<I", 0)
        except (TimeoutError, EOFError):
            ready = False
        if worker is not self._worker:
            return
        if ready:
            self._ready = True
            announce("device gpu")
        else:
            self._fail(worker, "the GPU worker did not start")

    def __call__(self, tokens, attention_mask):
        import numpy
        import torch

        worker = self._worker
        if not self._ready or worker is None:
            return None
        ids = tokens.numpy().astype(numpy.int32)[0]
        mask = attention_mask.numpy().astype(numpy.uint8)[0]
        try:
            worker.stdin.write(struct.pack("<I", len(ids)) + ids.tobytes() + mask.tobytes())
            worker.stdin.flush()
            rows, labels = struct.unpack("<II", read_exactly(worker.stdout, 8, self.ANSWER_SECONDS))
            if rows == 0xFFFFFFFF:
                error = read_exactly(worker.stdout, labels, self.ANSWER_SECONDS).decode(errors="replace")
                self._fail(worker, f"the GPU failed: {error}")
                return None
            data = read_exactly(worker.stdout, 4 * rows * labels, self.ANSWER_SECONDS)
        except TimeoutError:
            self._fail(worker, f"the GPU gave no answer within {self.ANSWER_SECONDS:g} s")
            return None
        except (EOFError, OSError) as error:
            self._fail(worker, f"the GPU worker stopped ({type(error).__name__})")
            return None
        self._failures = 0
        logits = numpy.frombuffer(data, dtype=numpy.float32).reshape(1, rows, labels)
        return torch.from_numpy(logits.copy())

    def _fail(self, worker: subprocess.Popen, reason: str) -> None:
        self._ready = False
        self._worker = None
        kill_group(worker)
        # A hung process can take a while to go; nothing waits for it here.
        threading.Thread(target=worker.wait, daemon=True).start()
        self._failures += 1
        delay = min(self.FIRST_RETRY_SECONDS * 2 ** (self._failures - 1), self.LAST_RETRY_SECONDS)
        announce(f"fallback {reason}; the CPU answers, the GPU is tried again in {delay:g} s")
        announce("device cpu")
        self._retry_after(delay)

    def _retry_after(self, delay: float) -> None:
        timer = threading.Timer(delay, self._retry, args=(delay,))
        timer.daemon = True
        timer.start()

    def _retry(self, delay: float) -> None:
        if available_mb() < self.RETRY_LEAST_AVAILABLE_MB:
            self._retry_after(delay)
            return
        self.start()

    def stop(self) -> None:
        if self._worker is not None:
            kill_group(self._worker)


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def available_mb() -> int:
    """The kernel's estimate of memory available without swapping, in MB."""
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def read_exactly(stream, size: int, seconds: float) -> bytes:
    """`size` bytes from a pipe, or TimeoutError once `seconds` have passed."""
    deadline = time.monotonic() + seconds
    chunks: list[bytes] = []
    remaining = size
    fd = stream.fileno()
    while remaining > 0:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise TimeoutError
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Forward:
    """The forward pass opf's pipeline calls: the GPU when it answers, else the CPU.

    The CPU model is loaded at start even with a GPU model, though it costs
    about 2.2 GB: the GPU fails when memory is short, the worst time to load
    it. It also answers while the GPU worker starts.
    """

    def __init__(self, gpu: GpuForward | None) -> None:
        self._gpu = gpu
        self._cpu = OnnxForward(ONNX_MODEL)

    def __call__(self, tokens, attention_mask):
        logits = self._gpu(tokens, attention_mask) if self._gpu is not None else None
        if logits is not None:
            return logits
        return self._cpu(tokens, attention_mask)


class Detector:
    def __init__(self) -> None:
        self._loaded = threading.Event()
        self._error: str | None = None
        self._lock = threading.Lock()
        self._runtime = None
        self._decoder = None
        self.gpu: GpuForward | None = None

    def load(self) -> None:
        announce("loading")
        try:
            import torch
            from opf._common.label_space import resolve_label_space_from_config
            from opf._core.decoding import build_sequence_decoder
            from opf._core.runtime import InferenceRuntime
            from opf._core.sequence_labeling import build_label_info

            # torch only runs opf's decoding here: thousands of tiny ops, which
            # a thread pool spread over performance and efficiency cores slows
            # down many times over (352 characters: 0.8 s on all cores, 0.05 s
            # on one). The forward pass has onnxruntime's own threads.
            torch.set_num_threads(1)
            announce("device cpu")
            if GPU_MODEL.is_file():
                announce("gpu starting")
                self.gpu = GpuForward(GPU_MODEL)
                self.gpu.start()
            config, encoding = load_encoding(MODEL_DIR)
            _, _, class_names = resolve_label_space_from_config(config, context=str(MODEL_DIR))
            label_info = build_label_info(class_names)
            self._runtime = InferenceRuntime(
                checkpoint=str(MODEL_DIR),
                model=Forward(self.gpu),
                encoding=encoding,
                label_info=label_info,
                device=torch.device("cpu"),
                # opf's own window on CPU; the attention band is 257 tokens, so
                # the window size changes speed, not accuracy.
                n_ctx=4096,
                trim_span_whitespace=True,
                discard_overlapping_predicted_spans=False,
                output_mode="typed",
                active_encoding_name=str(config["encoding"]),
                pad_token_id=int(encoding.eot_token),
                bidirectional_context=bool(config["bidirectional_context"]),
                category_version="",
            )
            self._decoder, _ = build_sequence_decoder(
                decode_mode="viterbi",
                label_info=label_info,
                viterbi_calibration_path=None,
                checkpoint_dir=str(MODEL_DIR),
            )
            announce("ready")
        except Exception as error:  # noqa: BLE001 - reported to the client
            self._error = f"{type(error).__name__}: {error}"
            announce(f"failed {self._error}")
        finally:
            self._loaded.set()

    def detect(self, text: str) -> list[dict[str, object]]:
        from opf._core.runtime import predict_text

        self._loaded.wait()
        if self._runtime is None:
            raise RuntimeError(self._error or "model not loaded")
        with self._lock:
            result = predict_text(self._runtime, text, decoder=self._decoder)
        return [
            {"label": label, "start": utf16_length(text[:start]), "end": utf16_length(text[:end])}
            for start, end, label in locate(text, result)
        ]


def locate(text: str, result) -> list[tuple[int, int, str]]:
    """Spans as offsets into `text`.

    When the tokenizer round trip changed the text, the model's offsets point
    into the decoded text; each span's own text is then found in the original,
    in order, and a span that cannot be found is dropped.
    """
    if not result.decoded_mismatch:
        return [(s.start, s.end, s.label) for s in result.spans]
    located: list[tuple[int, int, str]] = []
    cursor = 0
    for span in result.spans:
        start = text.find(span.text, cursor)
        if start < 0:
            continue
        end = start + len(span.text)
        located.append((start, end, span.label))
        cursor = end
    return located


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


class UnixHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = "ccpii"
        self.server_port = 0


def make_handler(detector: Detector):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/detect":
                self.reply(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                texts = json.loads(self.rfile.read(length))["texts"]
                spans = [detector.detect(text) for text in texts]
            except Exception as error:  # noqa: BLE001 - reported to the client
                self.reply(500, {"error": f"{type(error).__name__}: {error}"})
                return
            self.reply(200, {"spans": spans})

        def reply(self, status: int, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            # No access log: stderr is kept for what went wrong.
            pass

    return Handler


def serve() -> int:
    if not is_installed():
        announce("missing")
        return 0
    # Under /tmp, not $TMPDIR: a socket path has to stay under about 100 bytes.
    directory = tempfile.mkdtemp(prefix="ccpii.", dir="/tmp")
    socket_path = os.path.join(directory, "filter.sock")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    detector = Detector()
    try:
        # mkdtemp's directory is the owner's alone (0700), and so the socket in it.
        server = UnixHTTPServer(socket_path, make_handler(detector))
        announce(f"listening {socket_path}")
        threading.Thread(target=detector.load, daemon=True).start()
        server.serve_forever()
    finally:
        if detector.gpu is not None:
            detector.gpu.stop()
        shutil.rmtree(directory, ignore_errors=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if args.download:
        return download()
    return serve()


if __name__ == "__main__":
    sys.exit(main())
