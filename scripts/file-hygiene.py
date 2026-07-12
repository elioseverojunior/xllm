#!/usr/bin/env python3
"""Portable file-hygiene checks — the lefthook replacement for the
`pre-commit/pre-commit-hooks` repo (v6.0.0).

pre-commit-hooks ships these checks as small Python programs; this script
reimplements the ones the project used in a single stdlib-only file that
lefthook drives with `{staged_files}`. One file, zero dependencies, same intent.

Per-file checks run in parallel via a ThreadPoolExecutor for speed on large
staged-file sets.

Checks (validators — always block on violation):
    check-added-large-files   (--maxkb=1000)
    check-merge-conflict
    check-case-conflict
    check-symlinks
    check-executables-have-shebangs
    check-toml
    check-json
    forbid-new-submodules
Fixers (rewrite the file in place when --fix is given; otherwise reported):
    trailing-whitespace       (markdown hard-breaks preserved on *.md)
    end-of-file-fixer
    mixed-line-ending         (--fix=lf)

Intentionally NOT reimplemented — the project already handles these elsewhere:
    check-yaml          -> yamllint job (stricter, multi-document aware)
    detect-private-key  -> gitleaks job (broader secret scanning)
    detect-secrets      -> gitleaks job

Usage:
    file-hygiene.py [--fix] [paths...]   # paths come from lefthook {staged_files}
With no paths, there is nothing staged to check, so it exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MAX_KB = 1000
IGNORE_PREFIXES = ()

CONFLICT_MARKERS = (b"<<<<<<< ", b"======= ", b">>>>>>> ", b"|||||||")
CONFLICT_EXACT = (b"=======", b"<<<<<<<", b">>>>>>>")


class Report:
    """Accumulates violations and (in --fix mode) the set of rewritten files.
    Thread-safe: per-file results are collected via check_one_file() return
    values, then merged into this struct synchronously."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.fixed: set[str] = set()


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def ignored(rel: str) -> bool:
    return any(rel.startswith(p) for p in IGNORE_PREFIXES)


def _declared_submodule_paths() -> set[str]:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "config", "--file", ".gitmodules",
             "--get-regexp", r"^submodule\..*\.path$"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {parts[1] for line in out.splitlines()
            if len(parts := line.split(None, 1)) == 2}


# --------------------------------------------------------------------------- #
# Pure validators (no side effects, operate on data in memory)
# --------------------------------------------------------------------------- #
def _check_large_file(path: Path, rel: str) -> str | None:
    kb = path.stat().st_size / 1024
    if kb > MAX_KB:
        return f"file is {kb:.0f} KB, exceeds the {MAX_KB} KB limit"
    return None


def _check_merge_conflict(rel: str, data: bytes) -> str | None:
    for n, line in enumerate(data.splitlines(), 1):
        if line.startswith(CONFLICT_MARKERS) or line.rstrip() in CONFLICT_EXACT:
            return f"merge-conflict marker on line {n}"
    return None


def _check_symlink(path: Path, rel: str) -> str | None:
    if path.is_symlink() and not path.exists():
        return "broken symlink (points to a missing target)"
    return None


def _check_executable_shebang(path: Path, rel: str, data: bytes) -> str | None:
    if path.is_symlink() or is_binary(data):
        return None
    if path.stat().st_mode & 0o100 and not data.startswith(b"#!"):
        return "executable file is missing a shebang (#!) — or drop the +x bit"
    return None


def _check_toml(path: Path, rel: str) -> str | None:
    try:
        with path.open("rb") as fh:
            tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        return f"invalid TOML: {exc}"
    return None


def _check_json(path: Path, rel: str) -> str | None:
    try:
        json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"invalid JSON: {exc}"
    return None


# --------------------------------------------------------------------------- #
# Fixers — return the corrected bytes (or None if already clean).
# --------------------------------------------------------------------------- #
def _fix_line_endings(data: bytes) -> bytes | None:
    new = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return new if new != data else None


def _fix_trailing_whitespace(data: bytes, is_md: bool) -> bytes | None:
    out_lines = []
    for line in data.split(b"\n"):
        stripped = line.rstrip(b" \t")
        if is_md and line.endswith(b"  ") and line.strip():
            stripped = stripped + b"  "
        out_lines.append(stripped)
    new = b"\n".join(out_lines)
    return new if new != data else None


def _fix_end_of_file(data: bytes) -> bytes | None:
    if not data:
        return None
    new = data.rstrip(b"\n") + b"\n"
    return new if new != data else None


# --------------------------------------------------------------------------- #
# Per-file entry point (runs in a worker thread)
# --------------------------------------------------------------------------- #
def check_one_file(path: Path, rel: str, do_fix: bool
                   ) -> tuple[list[str], str | None]:
    """Inspect a single staged file.

    Returns (errors, fixed_path).  ``fixed_path`` is the relative path string
    when the file was rewritten (``do_fix=True``), or ``None`` otherwise.
    """
    errors: list[str] = []

    sym_err = _check_symlink(path, rel)
    if sym_err is not None:
        errors.append(f"  {rel}: {sym_err}")

    large_err = _check_large_file(path, rel)
    if large_err is not None:
        errors.append(f"  {rel}: {large_err}")

    try:
        data = path.read_bytes()
    except OSError:
        return errors, None

    sheb_err = _check_executable_shebang(path, rel, data)
    if sheb_err is not None:
        errors.append(f"  {rel}: {sheb_err}")

    if path.suffix == ".toml":
        toml_err = _check_toml(path, rel)
        if toml_err is not None:
            errors.append(f"  {rel}: {toml_err}")

    if path.suffix == ".json":
        json_err = _check_json(path, rel)
        if json_err is not None:
            errors.append(f"  {rel}: {json_err}")

    if is_binary(data):
        return errors, None

    mc_err = _check_merge_conflict(rel, data)
    if mc_err is not None:
        errors.append(f"  {rel}: {mc_err}")

    # Fixers
    is_md = rel.endswith(".md")
    new = data
    for fixer in (
        lambda d: _fix_line_endings(d),
        lambda d: _fix_trailing_whitespace(d, is_md),
        lambda d: _fix_end_of_file(d),
    ):
        result = fixer(new)
        if result is not None:
            new = result
    if new != data:
        if do_fix:
            path.write_bytes(new)
            return errors, rel
        else:
            errors.append(
                f"  {rel}: whitespace/line-ending/EOF issue — "
                "run `mise run hygiene-fix`"
            )

    return errors, None


# --------------------------------------------------------------------------- #
def _check_case_conflict(paths: list[str]) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    for rel in paths:
        low = rel.lower()
        if low in seen and seen[low] != rel:
            errors.append(f"  {rel}: case-only collision with {seen[low]!r}")
        else:
            seen[low] = rel
    return errors


def _check_new_submodule() -> list[str]:
    import subprocess

    errors: list[str] = []
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--raw", "--diff-filter=A"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return errors
    declared = _declared_submodule_paths()
    for line in out.splitlines():
        if line.startswith(":") and line.split()[1] == "160000":
            path = line.split("\t", 1)[-1]
            if path not in declared:
                errors.append(
                    f"  {path}: new git submodule is forbidden "
                    "(not declared in .gitmodules)"
                )
    return errors


def main(argv: list[str]) -> int:
    args = argv[1:]
    do_fix = "--fix" in args
    paths = [a for a in args if a != "--fix"]
    if not paths:
        return 0

    rep = Report()

    # Global checks (not per-file)
    rep.errors.extend(_check_case_conflict(paths))
    rep.errors.extend(_check_new_submodule())

    # Per-file checks run in parallel
    n_workers = min(os.cpu_count() or 1, 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for rel in paths:
            if ignored(rel):
                continue
            path = Path(rel)
            if not path.exists() or not path.is_file():
                continue
            futures[pool.submit(check_one_file, path, rel, do_fix)] = rel

        for f in as_completed(futures):
            errs, fixed_rel = f.result()
            rep.errors.extend(errs)
            if fixed_rel is not None:
                rep.fixed.add(fixed_rel)

    if rep.fixed:
        print(f"file-hygiene: fixed {len(rep.fixed)} file(s):", file=sys.stderr)
        for f in sorted(rep.fixed):
            print(f"  {f}", file=sys.stderr)
    if rep.errors:
        print("file-hygiene: violations:", file=sys.stderr)
        print("\n".join(rep.errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
