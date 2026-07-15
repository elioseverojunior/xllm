# xllm Architecture

## Overview

xllm is a CPU-first LLM inference engine written in Rust, porting the
[llama.cpp](https://github.com/ggml-org/llama.cpp) design. It structures
functionality into independently versioned crates, maximising reusability and
compile-time isolation.

## Guiding principles

- **TDD**: Test first, implement, refactor.
- **CPU-first**: CPU inference first; GPU (CUDA/HIP) behind optional feature flags.
- **Security-first**: `thiserror` in libraries, `anyhow` in binaries, no
  unwrap/expect on external input.
- **No unsafe code**: The `unsafe_code = "deny"` lint is set workspace-wide.
  Zero `unsafe` blocks, functions, or traits are permitted.
- **KISS, DRY, YAGNI, TDA, SOLID**.
- **Nightly Rust**: Edition 2024, newest language features.

---

## Workspace structure

The root `Cargo.toml` is a virtual workspace. Eleven crates live under
`crates/`:

```mermaid
mindmap
  root((xllm workspace))
    xllm
      Umbrella lib
      Re-exports all sub-crates
    xllm-cli
      CLI binary
    xllm-tensor
      Tensor ops
      fp16 support
    xllm-ggml
      Compute graph
      Multi-threaded execution
    xllm-model
      GGUF reader
      Weight management
    xllm-tokenizer
      BPE / SentencePiece
    xllm-sampling
      Greedy, top-k, top-p
    xllm-context
      KV cache
      Inference context
    xllm-quantize
      Q4_0, Q5_1, Q8_0
    xllm-bitnet
      1-bit ternary kernels
      T-MAC approach
    xllm-train
      Training scaffold
```

---

## Crate dependency graph

Arrows point from dependant to dependency:

```mermaid
flowchart BT
    xllm_cli["xllm-cli"]
    xllm["xllm (umbrella)"]
    xllm_train["xllm-train"]
    xllm_context["xllm-context"]
    xllm_model["xllm-model"]
    xllm_quantize["xllm-quantize"]
    xllm_bitnet["xllm-bitnet"]
    xllm_sampling["xllm-sampling"]
    xllm_tokenizer["xllm-tokenizer"]
    xllm_ggml["xllm-ggml"]
    xllm_tensor["xllm-tensor"]

    xllm_cli --> xllm
    xllm --> xllm_train
    xllm --> xllm_context
    xllm --> xllm_model
    xllm --> xllm_quantize
    xllm --> xllm_bitnet
    xllm --> xllm_sampling
    xllm --> xllm_tokenizer
    xllm --> xllm_ggml
    xllm --> xllm_tensor
    xllm_train --> xllm_context
    xllm_train --> xllm_model
    xllm_train --> xllm_ggml
    xllm_train --> xllm_tensor
    xllm_context --> xllm_model
    xllm_context --> xllm_ggml
    xllm_context --> xllm_tensor
    xllm_model --> xllm_ggml
    xllm_model --> xllm_tensor
    xllm_quantize --> xllm_tensor
    xllm_bitnet --> xllm_tensor
    xllm_sampling --> xllm_tensor
    xllm_ggml --> xllm_tensor
```

`xllm-tensor` is the sole leaf crate -- every domain crate depends on it.

---

## Layered architecture

```mermaid
flowchart LR
    subgraph App
        CLI[("xllm-cli")]
    end

    subgraph Facade
        Lib[("xllm -- re-exports")]
    end

    subgraph Domain
        direction TB
        M[("xllm-model")]
        T[("xllm-tokenizer")]
        S[("xllm-sampling")]
        C[("xllm-context")]
        Q[("xllm-quantize")]
        B[("xllm-bitnet")]
        TR[("xllm-train")]
    end

    subgraph Backend
        GGML[("xllm-ggml -- compute graph")]
    end

    subgraph Foundation
        Tensor[("xllm-tensor -- core type")]
    end

    CLI --> Lib
    Lib --> Domain
    Lib --> Backend
    Lib --> Foundation
    Domain --> Backend
    Domain --> Foundation
    Backend --> Foundation
```

---

## Inference pipeline

The data flow through a single generation step:

```mermaid
sequenceDiagram
    participant CLI as xllm-cli
    participant Lib as xllm (lib)
    participant Tokenizer as xllm-tokenizer
    participant Context as xllm-context
    participant Model as xllm-model
    participant GGML as xllm-ggml
    participant Sampler as xllm-sampling

    CLI->>Lib: generate(prompt, params)
    Lib->>Tokenizer: encode(prompt)
    Tokenizer-->>Lib: token_ids[]
    loop auto-regressive step
        Lib->>Context: eval(token_ids)
        Context->>Model: load_layer(layer_idx)
        Model->>GGML: build_compute_graph(tensors)
        GGML-->>Model: logits
        Model-->>Context: layer_output
        Context->>Context: update_kv_cache()
        Context-->>Lib: logits
        Lib->>Sampler: sample(logits, params)
        Sampler-->>Lib: token_id
        Lib->>Tokenizer: decode(token_id)
        Tokenizer-->>Lib: text_fragment
        Lib-->>CLI: text_fragment
    end
```

---

## External dependencies

| Dependency | Used by | Purpose |
| ---------- | ------- | ------- |
| `half` 2 | `xllm-tensor` | fp16 (half-precision) tensor storage |
| `byteorder` 1 | `xllm-model`, `xllm-tokenizer` | Binary file format reading (GGUF, BPE) |
| `memmap2` 0.9 | `xllm-model` | Memory-mapped file I/O for model weights |
| `rayon` 1 | `xllm-ggml` | Multi-threaded parallel computation |

GPU backends (`cudarc`, `hip-rs`) will be added behind feature flags in
Phase 4.

---

## Implementation phases

```mermaid
gantt
    title xllm implementation roadmap
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m-%d

    section Milestones
    Phase 0 complete              :milestone, m0, 2026-10-01, 0d
    Phase 1 complete              :milestone, m1, 2027-01-01, 0d
    Phase 2 complete              :milestone, m2, 2027-04-01, 0d

    section Phase 0 -- Project setup
    Workspace scaffolding         :p0a, 2026-07-12, 40d
    CI / tooling                  :p0b, 2026-08-01, 60d

    section Phase 1 -- Foundation
    Tensor type + basic ops       :p1a, 2026-10-01, 90d
    Compute graph (ggml backend)  :p1b, 2026-11-01, 60d
    GGUF model reader             :p1c, 2026-12-01, 60d
    Tokenizer BPE / SentencePiece :p1d, 2026-11-01, 60d

    section Phase 2 -- Core inference
    KV cache + attention          :p2a, 2027-01-01, 60d
    Sampling strategies           :p2b, 2027-01-01, 30d
    CLI + text generation         :p2c, 2027-03-01, 30d
    Quantization Q4_0             :p2d, 2027-03-01, 60d

    section Phase 3 -- Performance
    Multi-threaded CPU ops        :p3a, 2027-04-01, 60d
    SIMD kernels                  :p3b, 2027-06-01, 90d

    section Phase 3b -- BitNet
    Ternary kernels               :p3b1, 2027-05-01, 60d
    LUT-based matmul              :p3b2, 2027-07-01, 60d

    section Phase 4 -- GPU
    CUDA backend                  :p4a, 2027-07-01, 90d
    HIP (AMD) backend             :p4b, 2027-07-01, 90d

    section Phase 5 -- Advanced
    LoRA / adapters               :p5a, 2027-10-01, 60d
    vLLM batching                 :p5b, 2027-10-01, 90d
    Training / fine-tuning        :p5c, 2028-01-01, 90d
    Server mode (HTTP API)        :p5d, 2028-01-01, 60d
```

---

## Key design decisions

| Decision | Rationale |
| -------- | --------- |
| Follow llama.cpp API | Reuse knowledge, tools, and model files from the ecosystem |
| CPU-first, GPU behind feature flags | Keep core simple; GPU is an optimisation layered on top |
| `thiserror` for lib, `anyhow` for bin | Idiomatic Rust error handling; library errors are domain types |
| Unsafe code forbidden | `unsafe_code = "deny"` workspace-wide; zero unsafe code permitted |
| Nightly Rust | Edition 2024 features, async support, and newest language capabilities |
| GGUF file format | Full compatibility with the llama.cpp model ecosystem |
| BitNet b1.58 backend | Pure integer add/sub matmul -- no floating-point multiplications, massive CPU speedup |
| Virtual workspace | Independent versioning, faster compilation, cleaner dependency isolation |

---

## Configuration

All shared metadata lives in the root `Cargo.toml`:

- `[workspace.package]` -- edition, license, version.
- `[workspace.lints.rust]` -- lint policy (warnings-as-errors).
- `[workspace.dependencies]` -- shared external and internal dependencies.

Build profiles and target-specific linker flags are in `.cargo/config.toml`.
Rust toolchain is pinned via `rust-toolchain.toml` (stable).

---

## Testing strategy

- **Unit tests**: Inline `#[cfg(test)] mod tests` in every crate.
- **Integration tests**: End-to-end model loading and inference.
- **Benchmarks**: `cargo bench` for performance-sensitive operations.
- **Property-based**: `proptest` or `quickcheck` for tensor operations.
- **Fixtures**: Small test models (tiny GGUF, tokenizer test data).
