# xllm

CPU-first LLM inference engine — Rust port of
[llama.cpp](https://github.com/ggml-org/llama.cpp).

This is the umbrella crate that re-exports all xllm sub-crates for
convenient use. For the CLI binary, see `xllm-cli`.

## Crates

| Crate | Description |
| ----- | ----------- |
| `xllm-tensor` | Tensor operations |
| `xllm-ggml` | Compute graph backend |
| `xllm-model` | Model loading (GGUF) |
| `xllm-tokenizer` | BPE / SentencePiece |
| `xllm-sampling` | Token sampling strategies |
| `xllm-context` | KV cache / inference context |
| `xllm-quantize` | Quantization formats |
| `xllm-bitnet` | 1-bit ternary kernels |
| `xllm-train` | Training support |

## License

MIT OR Apache-2.0
