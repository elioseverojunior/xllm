# xllm-cli

Command-line interface for the xllm inference engine.

Downloads GGUF models from HuggingFace Hub and runs text generation with
configurable sampling parameters.

## Commands

### `download`

Fetch a GGUF model from HuggingFace Hub and cache it locally.

```sh
# Download the first .gguf file found in a repo
xllm download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

# Download a specific file
xllm download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --filename tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# Download to a custom directory
xllm download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --output-dir /tmp/models

# Force re-download even if the file exists locally
xllm download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --force
```

Models are cached to `~/.cache/huggingface/hub/` by default, matching the
HuggingFace Hub convention.

### `run`

Load a GGUF model and run text generation.

```sh
# Basic usage
xllm run --model model.gguf --prompt "Hello, world"

# Custom generation parameters
xllm run --model model.gguf \
  --prompt "Once upon a time" \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.95 \
  --seed 42
```

Generated tokens are streamed to stdout as they are produced.

## Typical workflow

```sh
# 1. Download a model
xllm download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

# 2. Run inference
xllm run \
  --model ~/.cache/huggingface/hub/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  --prompt "Hello"
```

## License

MIT OR Apache-2.0
