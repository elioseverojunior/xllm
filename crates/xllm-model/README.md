# xllm-model

Model loading, weight management, and GGUF format reader.

Reads GGUF model files (compatible with llama.cpp), manages model weights
as tensors, and exposes model architecture metadata. Uses memmap2 for
efficient file access.

## License

MIT OR Apache-2.0
