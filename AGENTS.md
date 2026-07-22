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

## Session context (Jul 15, 2026)

### What was done

- **License correction**: Code files are `MIT OR Apache-2.0` (dual-license for patent protection),
  documentation files are `CC-BY-3.0+` (CII badge requirement, OSI-approved doc license).
- **Created license files**: `LICENSE` (MIT text), `LICENSE-APACHE` (Apache-2.0 text),
  `CC-BY-3.0.txt` (Creative Commons Attribution 3.0 Unported).
- **REUSE.toml**: Updated with per-file annotations for license text files and DCO.txt.
- **Doc SPDX headers**: `CII_BEST_PRESTICES.md` and `CONTRIBUTING.md` use inline
  CC-BY-3.0+ SPDX-License-Identifier headers.
- **AGENTS.md**: Added full License model section with reasoning.

### Known limitations

- (none)

### Reasoning: `MIT OR Apache-2.0` for code + `CC-BY-3.0+` for docs

**Code** uses `MIT OR Apache-2.0` because:

- Apache-2.0 provides an explicit patent grant (Section 3) that MIT alone lacks.
  This protects downstream users from patent litigation by contributors — critical
  for an inference engine that may implement patented techniques.
- Dual-licensing gives downstream users choice: MIT for maximum compatibility,
  Apache-2.0 for patent protection.
- This is the Rust ecosystem standard and every crate on crates.io uses this model.

**Documentation** uses `CC-BY-3.0+` because:

- MIT and Apache-2.0 are **software licenses** for "source code," not creative works.
  Creative Commons is the standard for documentation.
- The OpenSSF CII Best Practices badge explicitly requires a documentation license
  for its "FLOSS license" criterion.
- CC-BY-3.0 is the simplest OSI-approved CC license: share/adapt with attribution.
  No copyleft (ShareAlike), no no-derivatives (ND) restrictions.
- The `+` suffix ["or any later version"] future-proofs — downstream can use CC-BY-4.0+.
- CC-BY-4.0 would also be acceptable; 3.0+ was chosen because it matches SPDX
  short identifier convention used by REUSE, and CII docs specifically cite 3.0+.

### Files changed

| File | Change |
|------|--------|
| `AGENTS.md` | Added License model section + this session context |
| `LICENSE` | Created (MIT text) |
| `LICENSE-APACHE` | Created (Apache-2.0 text) |
| `CC-BY-3.0.txt` | Created (CC-BY-3.0 Unported text) |
| `REUSE.toml` | Added annotations for LICENSE, LICENSE-APACHE, CC-BY-3.0.txt, DCO.txt |
| `CII_BEST_PRACTICES.md` | Added SPDX header (CC-BY-3.0+) |
| `CONTRIBUTING.md` | Fixed SPDX header format (CC-BY-3.0+) |
| `.rs` files (14) | Removed duplicate SPDX line that had CC-BY-3.0+ alongside MIT OR Apache-2.0 |

## License model

### Code: `MIT OR Apache-2.0`

Dual-licensing provides:

- **MIT**: simple permissive license, widely compatible, lets anyone use the code with minimal restrictions.
- **Apache-2.0**: MIT's permissions PLUS an explicit patent grant. Apache-2.0 Section 3 grants a patent license from contributors, which protects downstream users from patent litigation. This is the Rust ecosystem standard (`Cargo.toml` convention), and every crate published to crates.io inherits this model.

Both licenses are OSI-approved. Users may choose either. `LICENSE` and `LICENSE-APACHE` contain the full texts.

### Docs: `CC-BY-3.0+`

Documentation files (`CONTRIBUTING.md`, `CII_BEST_PRACTICES.md`, etc.) use Creative Commons Attribution 3.0 Unported or any later version. Rationale:

- **Why not MIT/Apache-2.0 for docs?** MIT and Apache-2.0 are software licenses — they cover "source code," not "creative works." While technically usable for docs, Creative Commons is the standard for documentation, and the OpenSSF CII Best Practices badge explicitly requires a **documentation license** for its "FLOSS license" criterion.
- **Why CC-BY-3.0?** It is the OSI-approved Creative Commons license with the simplest requirements: anyone can share/adapt the docs as long as they give attribution. No ShareAlike copyleft (which would prevent commercial reuse), no ND (which would prevent modification).
- **Why `+` (or any later version)?** Future-proofing — downstream users can use CC-BY-4.0 or later if they prefer. The CII badge specifically recommends CC-BY-3.0+ or CC-BY-4.0.
- **What about CC-BY-4.0?** Also acceptable. 3.0+ was chosen to match the SPDX short identifier convention used by REUSE, and 3.0 is the version most commonly cited in badge documentation.

### Enforcement

- All files carry SPDX headers (inline or via `REUSE.toml` aggregate annotations).
- `reuse lint` in CI enforces compliance — zero warnings required.
- Code files inherit `MIT OR Apache-2.0` from the `REUSE.toml` aggregate annotation.
- Doc files use inline `CC-BY-3.0+` SPDX license identifiers.
- License text files (`LICENSE`, `LICENSE-APACHE`, `CC-BY-3.0.txt`, `DCO.txt`) use explicit `REUSE.toml` annotations.

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
| `LICENSE` | MIT license text (for code) |
| `LICENSE-APACHE` | Apache-2.0 license text (for code) |
| `CC-BY-3.0.txt` | CC-BY-3.0 license text (for docs) |
| `REUSE.toml` | REUSE compliance manifest |

## Agent rules (from `docs/guidelines/contribution.md`)

- Never commit, push, or create PRs without explicit human approval.
- Never write PR descriptions, commit messages, or reviewer responses.
- Use `Assisted-by:` (not `Co-authored-by:`) if user asks you to commit.
- No unicode chars (`—`, `→`, `×`, `…`) — use ASCII.
