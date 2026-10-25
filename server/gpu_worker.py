# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openvino==2026.4.1",
#   "numpy",
# ]
# ///
"""The privacy filter's forward pass on an Intel GPU, for the ccpii sidecar.

The model is an OpenVINO conversion of the filter (GPU.md). The sidecar
starts this as a child of its own, so that a GPU that stops answering costs
a killed process rather than the sidecar: with system memory short, i915
has been seen to hang a compute context, and the process waiting on it then
ignores every signal but SIGKILL.

Usage: gpu_worker.py <model.xml>

The protocol is binary, on stdin and stdout (anything else goes to stderr):

- once the model is compiled and every input size has run once:
  `<u32 0>` (ready)
- request: `<u32 n>` then n int32 token ids and n uint8 attention flags
- answer: `<u32 n><u32 labels>` then n × labels float32 logits, or
  `<u32 0xFFFFFFFF><u32 length>` then a UTF-8 error, after which the worker
  exits, as the GPU context is not trusted again

End of stdin ends the worker.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

# Inputs are padded up to a multiple of this, so the GPU sees few distinct
# shapes: each new one costs about half a second the first time.
BUCKET = 256
# The sidecar's window (opf's n_ctx on CPU).
LONGEST = 4096
PAD_TOKEN = 199999
FAILED = 0xFFFFFFFF


def main() -> int:
    import numpy
    import openvino

    # The protocol owns the real stdout; a library printing there would break it.
    answers = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    requests = sys.stdin.buffer

    model = Path(sys.argv[1])
    core = openvino.Core()
    core.set_property("GPU", {"CACHE_DIR": str(model.parent / "cache")})
    request = core.compile_model(str(model), "GPU", {"INFERENCE_PRECISION_HINT": "f16"}).create_infer_request()

    def infer(ids, mask):
        n = ids.shape[1]
        size = -(-n // BUCKET) * BUCKET
        ids = numpy.pad(ids, ((0, 0), (0, size - n)), constant_values=PAD_TOKEN)
        mask = numpy.pad(mask, ((0, 0), (0, size - n)))
        return request.infer({"input_ids": ids, "attention_mask": mask})[0][0, :n]

    for size in range(BUCKET, LONGEST + 1, BUCKET):
        infer(numpy.zeros((1, size), dtype=numpy.int64), numpy.ones((1, size), dtype=numpy.int64))
    answers.write(struct.pack("<I", 0))
    answers.flush()

    while True:
        header = requests.read(4)
        if len(header) < 4:
            return 0
        (n,) = struct.unpack("<I", header)
        ids = numpy.frombuffer(requests.read(4 * n), dtype=numpy.int32).astype(numpy.int64)[None]
        mask = numpy.frombuffer(requests.read(n), dtype=numpy.uint8).astype(numpy.int64)[None]
        try:
            logits = numpy.ascontiguousarray(infer(ids, mask), dtype=numpy.float32)
        except Exception as error:  # noqa: BLE001 - reported to the sidecar
            # OpenVINO's messages run to many lines; the last one says what failed.
            last = (str(error).strip().splitlines() or [""])[-1]
            message = f"{type(error).__name__}: {last}".encode()
            answers.write(struct.pack("<II", FAILED, len(message)) + message)
            answers.flush()
            return 1
        answers.write(struct.pack("<II", *logits.shape) + logits.tobytes())
        answers.flush()


if __name__ == "__main__":
    sys.exit(main())
