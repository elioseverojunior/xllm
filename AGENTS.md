# xllm — AGENTS.md

## Project identity

- **xllm**: Rust port of [llama.cpp](https://github.com/ggml-org/llama.cpp),
  CPU-first LLM inference, GPU support (AMD/NVIDIA) later.
- Virtual workspace with 11 crates under `crates/`: `xllm` (umbrella lib),
  `xllm-cli` (binary), and 9 domain crates (tensor, ggml, model, tokenizer,
  sampling, context, quantize, bitnet, train).

## Toolchain

- **Nightly Rust** pinned via `rust-toolchain.toml`.
- Warnings-as-errors enforced via `[workspace.lints]` in root `Cargo.toml` + `.cargo/config.toml`.
- `rustfmt.toml` configures import grouping: std → external crates → local (`crate::`, `super::`).
- `.markdownlint.jsonc` controls markdown formatting (MD022, MD031, MD032 enforced).

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
  pnpm lint:md                  # markdown lint
  pnpm fix:md                   # auto-fix markdown issues
  ```

- Order: `cargo sort` → `taplo format Cargo.toml crates/*/Cargo.toml` → `pnpm fix:md` →
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

## Key files

| File | Purpose |
| --- | --- |
| `rust-toolchain.toml` | Pinned nightly channel |
| `.cargo/config.toml` | Build/linker tuning, profile optimization |
| `rustfmt.toml` | Import grouping: std, external, local last |
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
| `instructions.md` | Owner's vision: CPU-first, GPU later, nightly, security, TDD |
| `docs/guidelines/contribution.md` | AI agent rules — read before committing/pushing/PRing |
| `docs/plan/XLLM_PLAN.md` | Architecture plan |

## Agent rules (from `docs/guidelines/contribution.md`)

- Never commit, push, or create PRs without explicit human approval.
- Never write PR descriptions, commit messages, or reviewer responses.
- Use `Assisted-by:` (not `Co-authored-by:`) if user asks you to commit.
- No unicode chars (`—`, `→`, `×`, `…`) — use ASCII.
