# xllm — AGENTS.md

## Project identity

- **xllm**: Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp),
  CPU-first LLM inference, GPU support (AMD/NVIDIA) later.
- Virtual workspace with 11 crates under `crates/`: `xllm` (umbrella lib),
  `xllm-cli` (binary), and 9 domain crates (tensor, ggml, model, tokenizer,
  sampling, context, quantize, bitnet, train).

## Toolchain

- **Stable Rust** pinned via `rust-toolchain.toml`.
- Warnings-as-errors enforced via `[workspace.lints]` in root `Cargo.toml` + `.cargo/config.toml`.
- `rustfmt.toml` configures import reordering: std → external crates → local.
- `.rumdl.toml` controls markdown formatting.

## Workflow

- **TDD strictly**: write test → see red → implement → green → refactor. Never write production code first.
- **Commands**:

  ```sh
  cargo nextest run             # all tests (nextest)
  cargo nextest run <test_name>  # single test
  cargo clippy -- -D warnings   # Rust lint
  cargo sort                    # sort Cargo.toml dependencies
  cargo audit --deny warnings   # check security advisories
  cargo deny check              # check licenses, bans, sources
  cargo vet                     # supply chain vetting
  cargo machete                 # detect unused dependencies
  cargo miri test               # undefined behavior detection
  taplo format --check Cargo.toml crates/*/Cargo.toml  # TOML formatting check
  mise run markdownlint         # markdown lint
  mise run markdownlint:fix     # auto-fix markdown issues
  ```

- Order: `cargo sort` → `taplo format Cargo.toml crates/*/Cargo.toml` → `mise run markdownlint:fix` →
  `cargo fmt` → `cargo clippy -- -D warnings` → `cargo audit --deny warnings` →
  `cargo deny check` → `cargo vet` → `cargo machete` → `cargo nextest run`.
- Plan file `docs/plan/XLLM_PLAN.md` must be consulted/updated for architecture decisions.

## Worktrees

- Each feature branch gets its own worktree under `worktrees/` (ignored by git).

  ```sh
  git worktree add worktrees/<branch> <branch>
  ```

- **Reintegrate via rebase** to keep main linear:

  ```sh
  # from main worktree
  git pull --rebase
  git rebase main worktrees/<branch>
  git merge worktrees/<branch>   # fast-forward
  ```

- Never merge with `--no-ff` — main must stay a straight line.

## Architecture / design

- Follow llama.cpp's API and architecture patterns for compatibility.
- Design for CPU-first, keep GPU (CUDA/HIP) as an optional backend behind a feature flag.
- Security-first: use `thiserror` for errors in libraries, `anyhow` in binaries.
  Unsafe code is forbidden (`unsafe_code = "deny"`).
- Principles: TDD, KISS, DRY, YAGNI, TDA, SOLID.
- No comments unless explaining a non-obvious invariant.

## Session context (Jul 14, 2026)

### What was done

- **Download subcommand**: Added `xllm-cli download` to fetch GGUF models from HuggingFace Hub using reqwest.
- **Model caching**: Models saved to `~/.cache/huggingface/hub/` following HF conventions.
- **Smart file listing**: Auto-detects .gguf files via HF API (`/api/models/{repo}/tree/main`), picks first alphabetically.
- **Verified with real model**: Successfully downloaded `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` (483MB) from `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.
- **Clippy clean**: Fixed all warnings (collapsible if, unwrap_or_default, unused variable prefix with `_`).
- **opencode.json**: Updated permissions to `"ask"` for general bash, allowlisted cargo/pnpm/rustup/mise/mkdir/rm, specific git commands, denied git commit/push and gh actions.

### Known limitations

- Tensor type 14 (Q2_K) in GGUF not yet supported by `xllm-model` — need to extend tensor type enum.
- Sha256 verification skipped (would need HF API calls + auth).
- `--no-symlinks` flag accepted but not yet wired to logic.

### URLs

- HF model download: `https://huggingface.co/{repo}/resolve/main/{filename}`
- HF file list API: `https://huggingface.co/api/models/{repo}/tree/main`

## Key files

| File | Purpose |
| --- | --- |
| `rust-toolchain.toml` | Pinned stable channel |
| `.cargo/config.toml` | Build/linker tuning, profile optimization |
| `rustfmt.toml` | Import reorder: std, external, local last |
| `deny.toml` | License/bans/sources policy (cargo-deny) |
| `.cargo/audit.toml` | Advisory severity thresholds |
| `supply-chain/` | cargo-vet supply chain audits |
| `.taplo.toml` | TOML formatting rules, schema validation disabled |
| `Cargo.toml` | Virtual workspace root (`[workspace]` only) |
| `crates/xllm/Cargo.toml` | Umbrella library crate |
| `crates/xllm-cli/Cargo.toml` | CLI binary crate |
| `crates/xllm-tensor/Cargo.toml` | Tensor operations crate |
| `crates/xllm-ggml/Cargo.toml` | Compute graph crate |
| `crates/xllm-model/Cargo.toml` | Model loading crate |
| `crates/xllm-tokenizer/Cargo.toml` | Tokenizer crate |
| `crates/xllm-sampling/Cargo.toml` | Sampling strategies crate |
| `crates/xllm-context/Cargo.toml` | Inference context crate |
| `crates/xllm-quantize/Cargo.toml` | Quantization crate |
| `crates/xllm-bitnet/Cargo.toml` | BitNet kernels crate |
| `crates/xllm-train/Cargo.toml` | Training crate |
| `instructions.md` | Owner's vision: CPU-first, GPU later, security, TDD |
| `docs/guidelines/contribution.md` | AI agent rules — read before committing/pushing/PRing |
| `docs/plan/XLLM_PLAN.md` | Architecture plan |

## Agent rules (from `docs/guidelines/contribution.md`)

- Never commit, push, or create PRs without explicit human approval.
- Never write PR descriptions, commit messages, or reviewer responses.
- Use `Assisted-by:` (not `Co-authored-by:`) if user asks you to commit.
- No unicode chars (`—`, `→`, `×`, `…`) — use ASCII.
