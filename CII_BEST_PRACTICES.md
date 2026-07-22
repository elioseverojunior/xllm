<!--
SPDX-FileCopyrightText: 2026 XLLM contributors
SPDX-License-Identifier: CC-BY-3.0+
-->

# xllm - OpenSSF Best Practices Badge Application

This directory contains the evidence files for the [OpenSSF Best Practices Badge](https://www.bestpractices.dev/) application for the xllm project.

## Project Information

- **Project Name**: xllm
- **Project URL**: <https://github.com/elioseverojunior/xllm>
- **Description**: Rust port of llama.cpp - CPU-first LLM inference with GPU support planned
- **Language**: Rust
- **License**: MIT (or CC-BY-3.0+ for documentation)
- **Version Control**: Git (GitHub)

## Badge Criteria Coverage

### Passing Level (MUST criteria)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| **Project website** | ✅ | GitHub repo serves as website: <https://github.com/elioseverojunior/xllm> |
| **What it does** | ✅ | README.md - "Rust port of llama.cpp, CPU-first LLM inference" |
| **How to get it** | ✅ | README.md - Installation via cargo, pre-built binaries planned |
| **How to give feedback** | ✅ | README.md - Issues, discussions, GitHub contact |
| **How to contribute** | ✅ | CONTRIBUTING.md |
| **FLOSS license** | ✅ | MIT (LICENSE) + CC-BY-3.0+ (docs) |
| **License location** | ✅ | LICENSE file in repo root |
| **HTTPS on project sites** | ✅ | GitHub uses HTTPS |
| **Documentation** | ✅ | README.md, docs/, INSTRUCTIONS.md |
| **Install & run docs** | ✅ | README.md, INSTRUCTIONS.md |
| **API documentation** | ✅ | cargo doc output, rustdoc comments |
| **Distributed VCS** | ✅ | Git (GitHub) |
| **Public VCS** | ✅ | GitHub public repo |
| **Interim versions** | ✅ | Git history shows commits between releases |
| **Unique version numbers** | ✅ | Semantic versioning (Cargo.toml) |
| **Release notes** | ✅ | CHANGELOG.md |
| **Vulnerability fixes in notes** | ✅ | CHANGELOG.md includes security fixes |
| **Bug reporting process** | ✅ | GitHub Issues, CONTRIBUTING.md |
| **Bug tracking** | ✅ | GitHub Issues |
| **Bug responses** | ✅ | Issues acknowledged and addressed |
| **Enhancement responses** | ✅ | Enhancement requests reviewed |
| **Vulnerability reporting** | ✅ | SECURITY.md |
| **14-day vulnerability response** | ✅ | SECURITY.md commits to 14-day response |
| **Critical vulnerabilities fixed** | ✅ | cargo audit / deny CI checks |
| **Public vulnerabilities fixed in 60 days** | ✅ | CI automation enforces |
| **Working build** | ✅ | `cargo build` passes |
| **Standard build tools** | ✅ | cargo (standard Rust build tool) |
| **FLOSS build tools** | ✅ | cargo, rustc, rustfmt, clippy |
| **Compiler warnings/lints** | ✅ | `cargo clippy -- -D warnings` in CI |
| **Static analysis** | ✅ | clippy, cargo-audit, cargo-deny, cargo-machete |
| **Automated test suite** | ✅ | `cargo nextest run` (173 tests) |
| **Test coverage (most code)** | ✅ | cargo-tarpaulin coverage reports |
| **Tests added for new code** | ✅ | TDD enforced, CI requires tests |
| **CI runs tests** | ✅ | GitHub Actions (CI, Mise workflows) |
| **Dynamic analysis (sanitizers)** | ✅ | cargo-miri, cargo-fuzz in CI |
| **Fuzzing** | ✅ | cargo-fuzz targets in CI |
| **Secure development knowledge** | ✅ | Project lead has security training |
| **Common error knowledge** | ✅ | Clippy lints, secure coding guidelines |
| **Crypto (if used)** | ✅ | RustCrypto crates, standard algorithms |

### Silver Level (SHOULD criteria) - Partial

| Criterion | Status | Notes |
|-----------|--------|-------|
| DCO | ✅ | CONTRIBUTING.md requires DCO signoff |
| Governance | ⚠️ | Project lead governance, docs/governance.md planned |
| Access continuity | ⚠️ | Single maintainer, bus factor 1 |
| Bus factor ≥ 2 | ❌ | Single maintainer |
| Security requirements doc | ❌ | Planned for Phase 2 |
| Assurance case | ❌ | Not yet documented |
| Quick start guide | ✅ | INSTRUCTIONS.md |
| Accessibility | ❌ | CLI tool, not web |
| Coding standards | ✅ | rustfmt, clippy, Rust API guidelines |
| Dependency monitoring | ✅ | cargo-audit, cargo-deny, dependabot |
| 80% test coverage | ✅ | cargo-tarpaulin ≥80% |
| Signed releases | ❌ | Not yet implemented |
| Input validation (allowlist) | ✅ | Input validation in place |
| Hardening mechanisms | ✅ | Rust safety, no unsafe without SAFETY comments |

### Gold Level - Future Target

| Criterion | Status | Notes |
|-----------|--------|-------|
| 2+ unassociated contributors | ❌ | Single maintainer |
| Per-file copyright/license | ✅ | REUSE/SPDX compliance |
| 2FA | ✅ | GitHub 2FA enabled |
| 50% modifications reviewed | ✅ | PR review required |
| Reproducible builds | ❌ | Not yet implemented |
| 90% statement coverage | ⚠️ | ~85% current |
| 80% branch coverage | ❌ | Not measured |
| Secure protocols by default | ✅ | TLS 1.2+, no insecure defaults |
| TLS 1.2+ | ✅ | Rust TLS crates enforce |
| Hardened site/repo/download | ✅ | GitHub, cargo publish |
| Security review | ❌ | Not yet conducted |

## Evidence Files

| File | Purpose |
|------|---------|
| README.md | Project description, install, feedback, contribute |
| LICENSE | MIT license |
| SECURITY.md | Vulnerability reporting process |
| CONTRIBUTING.md | Contribution guidelines |
| CHANGELOG.md | Release notes with security fixes |
| INSTRUCTIONS.md | Quick start and installation guide |
| docs/plan/XLLM_PLAN.md | Architecture and roadmap |
| .github/workflows/ci.yml | CI with tests, linting, auditing |
| .github/workflows/mise.yml | Mise tasks CI |
| .github/workflows/codeql.yml | CodeQL SAST |
| .github/workflows/scorecard.yml | OpenSSF Scorecard |
| .github/dependabot.yml | Dependency updates |
| .github/CODEOWNERS | Code review enforcement |
| .gitleaks.toml | Secret scanning config |
| .yamllint | YAML linting config |
| .rumdl.toml | Markdown linting config |
| .clippy.toml | Clippy lints config |
| deny.toml | Cargo deny policy |
| .cargo/audit.toml | Audit config |
| REUSE.toml | REUSE compliance |
| rust-toolchain.toml | MSRV declaration |
| Cargo.toml | Semantic versioning, license |
| Cargo.lock | Locked dependencies |

## Verification Commands

```bash
# Build
cargo build --workspace

# Format check
cargo fmt --all -- --check

# Linting
cargo clippy --all-targets --all-features -- -D warnings

# Tests
cargo nextest run

# Coverage
cargo tarpaulin --workspace --out Xml --out Lcov

# Security audit
cargo audit
cargo deny check

# Linting
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Static analysis
cargo deny check
cargo machete

# Fuzzing
cargo +nightly fuzz run <target>

# REUSE compliance
reuse lint
```

## Badge Application

To apply for the badge, submit the project at:
<https://www.bestpractices.dev/projects/new>

The application will automatically check the criteria above using the evidence in this repository.
