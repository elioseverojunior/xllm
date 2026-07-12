# xllm — Architecture Plan

## Overview

xllm is a Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp),
a CPU-first LLM inference engine. It follows llama.cpp's API and architecture
patterns for compatibility, while adding safety guarantees via Rust's type
system.

## Guiding principles

- TDD: Test first, implement, refactor. Never production code before tests.
- CPU-first: Optimize for CPU inference first, GPU (CUDA/HIP) as optional
  feature flags.
- Security-first: No unwrap/expect on external input; `thiserror` in libraries,
  `anyhow` in binaries.
- KISS, DRY, YAGNI, TDA, SOLID.

## Module structure

The project is a virtual workspace with 11 crates under `crates/`:

```text
Cargo.toml                     # Virtual workspace root ([workspace] only)
crates/
  xllm/                        # Umbrella library (re-exports all sub-crates)
    src/lib.rs
  xllm-cli/                    # CLI entrypoint binary
    src/main.rs
  xllm-tensor/                 # Tensor operations (matmul, reshape, etc.)
    src/lib.rs
  xllm-ggml/                   # Compute graph / backend (matches llama.cpp API)
    src/lib.rs
  xllm-model/                  # Model loading, weights, architecture
    src/lib.rs
  xllm-tokenizer/              # Tokenizer (BPE, SentencePiece)
    src/lib.rs
  xllm-sampling/               # Token sampling strategies
    src/lib.rs
  xllm-context/                # Inference context, KV cache
    src/lib.rs
  xllm-quantize/               # Quantization (Q4_0, Q5_1, Q8_0, etc.)
    src/lib.rs
  xllm-bitnet/                 # 1-bit ternary kernels (BitNet b1.58)
    src/lib.rs
  xllm-train/                  # Training support (later)
    src/lib.rs
```

Each crate uses inline `#[cfg(test)] mod tests` for unit tests.

## Implementation phases

### Phase 1 — Foundation

- Tensor type with basic ops (matmul, add, reshape, slice).
- Compute graph abstraction (`ggml` backend).
- Model file format reader (GGUF).
- Tokenizer (BPE/SentencePiece).
- Single-precision CPU inference for a small model.

### Phase 2 — Core inference

- KV cache with attention.
- Sampling (greedy, top-k, top-p, temperature).
- CLI that loads a model and generates text.
- Quantization (Q4_0).

### Phase 3 — Performance

- Multi-threaded CPU ops.
- SIMD (x86 AVX2/AVX-512, ARM NEON, WASM SIMD).
- Memory optimizations (mmap, buffer reuse).

### Phase 3b — 1-bit inference (BitNet)

- Ternary weight format `{-1, 0, +1}` (1.58-bit) — zero floating-point multiplications,
  pure integer addition/subtraction.
- Lookup-table-based ternary matmul kernels (following T-MAC approach).
- GGUF model loading for BitNet b1.58 models.
- Shared with Phase 3: SIMD and multi-threading apply directly.

### Phase 4 — GPU support (optional features)

- CUDA backend behind `cuda` feature flag.
- HIP (AMD) backend behind `hip` feature flag.
- Device memory management, kernel launch.

### Phase 5 — Advanced

- LoRA / adapters.
- vLLM-style batching.
- Training / fine-tuning.
- Server mode (HTTP API).

## Build system & features

Shared dependencies are defined in the root `Cargo.toml` `[workspace.dependencies]`
and inherited by each crate. Feature flags are per-crate; the umbrella `xllm` crate
exposes the combined feature surface.

```toml
# Root Cargo.toml (workspace)
[workspace.dependencies]
half = "2"
byteorder = "1"
memmap2 = "0.9"
rayon = "1"

# Per-crate features (e.g. xllm-ggml)
[features]
default = ["cpu"]
cpu = []
cuda = ["dep:cudarc"]
hip = ["dep:hip-rs"]
```

## Testing strategy

- Unit tests inline (per module).
- Integration tests in `tests/` for end-to-end model loading and inference.
- Use small test models (e.g., 7B tokenizer test, tiny GGUF fixture).
- Benchmark tests (`cargo bench`) for performance-sensitive ops.
- Property-based testing (`proptest` or `quickcheck`) for tensor ops.

## Key design decisions

| Decision | Rationale |
| --- | --- |
| Follow llama.cpp API | Reuse knowledge, tools, and model files |
| CPU-first, GPU behind feature flags | Keep core simple; GPU is an optimization |
| `thiserror` for lib, `anyhow` for bin | Idiomatic Rust error handling |
| Unsafe code forbidden | `unsafe_code = "deny"` across the entire workspace; zero unsafe code allowed |
| Nightly Rust | Enable newest language features (edition 2024, async, etc.) |
| GGUF file format | Compatible with llama.cpp ecosystem |
| BitNet b1.58 backend | 1-bit ternary inference (add/sub only, no multiply) — massive CPU speedup |
