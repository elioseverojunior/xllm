# How to Test GGUF Model Download and CPU Inference with xllm

This guide explains how to download GGUF-format language models and test them with the xllm CPU inference engine.

## Overview

xllm supports GGUF (GPT-Generated Unified Format) model files, which is the same format used by llama.cpp. This allows you to use the same models with xllm that you would use with llama.cpp.

## Finding GGUF Models

### HuggingFace Model Library

The easiest way to find GGUF models is through HuggingFace:

- Visit: <https://huggingface.co/models?library=gguf&sort=downloads>
- Filter by "gguf" library tag
- Models are sorted by download count (most popular first)

### Popular GGUF Models

As of 2026, some popular GGUF models include:

| Model | Size | Quantization | Use Case |
|-------|------|--------------|----------|
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | Q4_K_M | Quick testing, low resources |
| Phi-3-mini-4k-instruct | 3.8B | Q4_K_M | Good quality, small footprint |
| Mistral-7B-Instruct-v0.3 | 7B | Q4_K_M | Balanced performance/quality |
| Llama-3-8B-Instruct | 8B | Q4_K_M | Latest Llama 3 capabilities |
| CodeLlama-7B-Python-v1.0 | 7B | Q4_K_M | Code generation specialized |

## Downloading a Model

### Method 1: Direct Download (Recommended for Testing)

For quick testing, download a small model directly:

```bash
# TinyLlama - excellent for initial testing (~600MB)
wget https://huggingface.co/ibm-research/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_K_M.gguf

# Or Phi-3-mini - better quality (~2GB)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

### Method 2: Using huggingface_hub Python Library

```bash
# Install if needed
pip install huggingface_hub huggingface-hub[hf-transfer]

# Download model
python -c "
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id='ibm-research/TinyLlama-1.1B-Chat-v1.0-GGUF',
    filename='tinyllama-1.1b-chat-v1.0.q4_K_M.gguf',
    local_dir='./models',
    local_dir_use_symlinks=False
)
print(f'Model downloaded to: {model_path}')
"
```

### Method 3: Using Git LFS (for larger models)

```bash
# Install git-lfs if needed
# Download and install from: https://git-lfs.github.com/

# Clone model repository
git lfs install
git clone https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
cd TinyLlama-1.1B-Chat-v1.0-GGUF
# The .gguf file will be in the directory
```

## Understanding Model Files

When you download a GGUF model, you'll typically get:

```text
model-name.q4_K_M.gguf
```

The filename indicates:

- **model-name**: The base model (e.g., tinyllama-1.1b-chat-v1.0)
- **q4**: Quantization method (4-bit)
- **K_M**: Specific quantization variant (K_M represents a good balance of quality/size)

Common quantization types (in order of quality→size):

- IQ2_XS, IQ2_S, IQ2_XXS (lowest quality, smallest)
- IQ3_XS, IQ3_S, IQ3_XXS
- IQ4_NL, IQ4_XS
- Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q4_K_L, Q5_K_S, Q5_K_M, Q5_K_L, Q6_K
- Q8_0 (highest quality, largest)
- F16 (float16, very large)
- F32 (float32, largest)

## Testing with xllm CLI

Once you have a GGUF model file, test it with:

```bash
# Basic usage
cargo run --bin xllm-cli -- /path/to/model.gguf \
  --prompt "Hello, how are you today?" \
  --max-tokens 50

# With different sampling parameters
cargo run --bin xllm-cli -- /path/to/model.gguf \
  --prompt "Explain quantum computing in simple terms:" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-p 0.9 \
  --top-k 40
```

### Sampling Parameters Explained

- `--prompt TEXT`: Input text to start generation
- `--max-tokens N`: Maximum number of tokens to generate (default: 128)
- `--temperature FLOAT`: Randomness (0.0 = deterministic, 1.0 = normal, >1.0 = more random)
- `--top-k INTEGER`: Limit sampling to top K tokens (0 = disabled)
- `--top-p FLOAT`: Nucleus sampling - sample from top P probability mass (1.0 = disabled)
- `--seed INTEGER`: Random seed for reproducibility (0 = random)

## Expected Output

When running successfully, you should see:

```text
Loading model from /path/to/model.gguf...
Model architecture: llama
Tensor count: 72
Vocabulary size: 32000
Context length: 2048
Embedding length: 4096

Hello, how are you today? I'm doing well, thank you for asking! How can I assist you today?

--- Generated 24 tokens ---
```

## Troubleshooting

### Common Issues

1. **"Error loading model"**
   - Verify the file is a valid GGUF file (not corrupted download)
   - Try re-downloading the model
   - Ensure you have read permissions to the file

2. **"Context length exceeded"**
   - Your prompt + `--max-tokens` exceeds the model's context length
   - Reduce `--max-tokens` or use a shorter prompt
   - Check model's context length in the startup output

3. **Slow performance on first run**
   - First inference includes kernel compilation and caching
   - Subsequent runs will be faster

4. **Out of memory**
   - Try a smaller model or more aggressive quantization (Q2_K, Q3_K_S)
   - Close other memory-intensive applications

## Working With Your Qwen Model

You mentioned having Qwen2.5-Coder-32B-Instruct-GGUF cached. This model appears to be sharded (split into multiple files):

```text
qwen2.5-coder-32b-instruct-q4_k_m-00001-of-00003.gguf
qwen2.5-coder-32b-instruct-q4_k_m-00002-of-00003.gguf
qwen2.5-coder-32b-instruct-q4_k_m-00003-of-00003.gguf
```

### Option 1: Combine Shards (Advanced)

You would need to use a tool to combine the shards into a single GGUF file. This typically requires:

1. Using llama.cpp's conversion tools with the original model weights
2. Or using specialized GGUF merging tools

### Option 2: Use a Non-Sharded Version

Look for the same model in a non-sharded format on HuggingFace, or use a different model size:

- Qwen2.5-Coder-7B-Instruct (more likely to have single-file GGUF versions)
- Qwen2.5-1.5B-Instruct
- Qwen2-0.5B-Instruct

## Next Steps

Once you've verified basic inference works:

1. **Experiment with different models** - Try various sizes and quantizations
2. **Adjust sampling parameters** - See how temperature, top-k, top-p affect output
3. **Try longer conversations** - Build up context over multiple interactions
4. **Explore specialized models** - Code generation, multilingual, instruction-following variants
5. **Prepare for Phase 2** - Once KV cache and attention are implemented, performance will improve significantly

## Verification

To confirm xllm is working correctly before downloading models:

```bash
# Run the comprehensive test suite
cargo nextest run

# Should show: 173 tests run: 173 passed, 0 skipped
```

This validates that all core components (tensor ops, compute graph, model loading, tokenizer, inference pipeline) are functioning correctly.
