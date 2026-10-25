# Running the filter on an Intel GPU

ccpii can run the model's forward pass on an Intel GPU with XMX matrix engines (Arc, including the Arc 130T/140T in Core Ultra 200H). On an Arc 140T, long texts ran about five times as fast as on the CPU: a 15 KB file in 0.75 s against 3.8 s. ccpii does not make the GPU model; this page says what it expects and how one was made.

## What ccpii expects

An OpenVINO model at `~/.cache/ccpii/privacy-filter/openvino-int8/model.xml` (with its `model.bin`), beside the CPU model `/ccpii install` puts in `~/.cache/ccpii/privacy-filter`. When it is there, the sidecar runs it in a worker process, `server/gpu_worker.py`, and says so: `/ccpii status` shows which device answers.

- **Inputs**: `input_ids` and `attention_mask`, both `int64`, shape `[1, n]` with `n` dynamic. Token ids are `o200k_base`, as opf tokenizes.
- **Output**: logits `[1, n, 33]` in the label order of the checkpoint's `config.json` (`O`, then `B-`/`I-`/`E-`/`S-` per category).
- **Runtime**: the worker pins `openvino==2026.4.1`, compiles for `GPU` with `INFERENCE_PRECISION_HINT=f16`, keeps a compiled-model cache in `openvino-int8/cache`, and pads every input with token `199999` and mask `0` up to a multiple of 256, at most 4096 (opf's window). It runs each of those 16 sizes once before it reports ready, so the GPU never compiles a new shape mid-request. Padding must not change the logits of the real tokens.

The CPU model stays loaded beside it, about 2.2 GB more. With system memory short, the i915 driver has been seen to hang the GPU's compute work: the call never returns, and the process ignores every signal but SIGKILL. So each GPU call gets 5 s; past that, or on an error, the worker's whole process group is killed and the CPU answers that call. A new worker starts after 30 s, twice as long after each failure in a row up to 10 minutes, and only once 4 GB are available, since a starting worker takes about 2 GB.

## Why it takes a conversion

- The ONNX exports in the model's repository use onnxruntime's own operators: `com.microsoft.MoE` or `QMoE` for the experts. OpenVINO's ONNX frontend converts neither, and onnxruntime's OpenVINO execution provider crashed on both GPU and NPU.
- The GPU plugin's SDPA takes no attention-sink input, and the model's attention has sinks.
- The plugin has a sparse MoE kernel, but it fuses only one graph shape into it, and only from compressed weights.

PyTorch's XPU build runs opf's own model, but its plain MoE gathers each token's expert weights and was four times slower than the CPU on long texts; opf's Triton MoE kernels, made to run on XPU, were faster but still slower than the CPU.

## How one was made

What worked, starting from transformers' `OpenAIPrivacyFilterForTokenClassification` (the `config.json` and `model.safetensors` at the top of the model's repository), traced by OpenVINO's PyTorch frontend.

1. **Experts as one dense block, in the shape the plugin fuses.** OpenVINO's `ConvertTiledMoeBlockTo2GatherMatmuls` matches: the tokens tiled once per expert (`Tile`, `Reshape`), one `MatMul` with `transpose_b` over `[experts, out, in]` gate-and-up weights plus a constant bias, gate and up cut apart by two `Slice`s, `Clamp` then `+ 1` on up, `Minimum` then `Swish` on gate, their product, a second `MatMul` with `transpose_b` plus bias, a `Reshape` to `[experts, tokens, 1, hidden]`, a multiply by the routing weights scattered into a dense `[tokens, experts]` matrix (`ScatterElementsUpdate`, `Transpose`, `Reshape`, `Unsqueeze`), and a `ReduceSum` over experts. Registered as an experts implementation (`transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`), this computes every expert on every token on the CPU, which only matters while tracing.
2. **Gate and up interleaved.** The fused kernel reads gate and up as alternating rows and takes the gate's position from the start of the step-2 slice. The checkpoint keeps them as two halves; slicing the halves with step 1 matches the pattern, fuses, and then gives wrong labels with no error. Interleave the rows of the gate-and-up weights and bias, and slice `[..., 0::2]` (gate) and `[..., 1::2]` (up).
3. **Sinks as one more key.** Widen the head size from 64 to 80 with zeros; give each query a 1 in column 64, each real key a 0 there, and add one key per head whose column 64 holds that head's sink, with a zero value vector. Its logit is the sink, exactly as in the original's concatenated logit, it adds nothing to the output, and the graph is plain SDPA. Extend the mask with a column every query may see. transformers passes no mask at all when the text is shorter than the 257-token sliding window, so handle that case.
4. **Trace with a dynamic length** (`input=[("input_ids", [1, -1], i64), ("attention_mask", [1, -1], i64)]`) and an example longer than the sliding window, so the traced graph computes the mask.
5. **Compress the weights to int8** with NNCF (`INT8_ASYM`, per channel). The plugin's MoE fusion takes only `u8`/`i8`/`u4`/`i4` weights, runs only with `f16` inference precision, and only on a GPU with XMX. Save without `compress_to_fp16`: an fp16 constant gets a `Convert` in front of it, and the pattern wants the biases and scalars as plain constants.
6. **Check it.** The compiled model's runtime graph (`compiled.get_runtime_model()`, each op's `layerType`) should hold `moe_gemm`, `moe_gather` and `moe_scatter_reduction`; plain `FullyConnected` and `Gemm` for the experts means it did not fuse and every expert runs on every token. Then compare labels with the original model on text that has personal data in it. Random token ids are labelled `O` almost everywhere, so they agree even with a broken model.

With torch 2.14.1, transformers 5.20.2, openvino 2026.4.1 and nncf 3.4.0, the rewritten model matched the original's logits to about 1e-5 before tracing, and the int8 model on the GPU labelled a sample sentence as the original does. On seven texts, spans differed from the unquantized model's in 3 places on the GPU and in 1 with ccpii's CPU model. The conversion took about 50 s and 15 GB of memory at its peak; the model is 1.4 GB.

The fusion is fragile across versions: each of the points above was found from a version where something else in the graph did not match, and other versions of transformers or OpenVINO may trace a different graph.
