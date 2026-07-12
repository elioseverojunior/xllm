#!/usr/bin/env python3
"""Update SHA pins for GitHub Actions to the latest release.

Scans every ``.github/workflows/*.yml`` and ``.github/actions/**/action.yml``
for ``uses:`` references to external actions and pins them to the commit SHA
of the latest public release.  Only updates within the same major version
(no automatic breaking-change upgrades): if the currently pinned version is
``v3.x`` and the latest is ``v4.y``, the script *reports* the available major
bump but does **not** rewrite it.  Branch-based refs (``@stable``,
``@nightly``) are resolved to the branch-tip SHA every time.

Requires ``GITHUB_TOKEN`` (or ``GH_TOKEN``) env var for authenticated API
requests — unauthenticated clients are rate-limited to 60 requests/hour,
which is quickly exhausted by a multi-repo scan.

Usage
-----
    python3 scripts/update-actions-pins.py          # check-only (dry-run)
    python3 scripts/update-actions-pins.py --write  # write changes
    python3 scripts/update-actions-pins.py --write --apply-major

Exit code
---------
0 if everything is up-to-date (or changes written).
1 if there are stale pins (dry-run), write failed, or API error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Actions that use branch refs instead of version tags.
# Key = repo, value = list of valid branch names to check.
BRANCH_ONLY_REPOS: dict[str, list[str]] = {
    "dtolnay/rust-toolchain": ["stable", "nightly"],
}

# Branch overrides — when a ref like ``@nightly`` is used, resolve that branch.
BRANCH_OVERRIDES: dict[str, str] = {
    "nightly": "nightly",
}

# Actions whose latest release tag doesn't follow semver.
# Key = repo, value = ref to resolve (tag or branch).
REF_OVERRIDES: dict[str, str] = {
    "github/codeql-action": "v3",
}

REUSABLE_PATHS: set[str] = {
    "slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml",
}


@dataclass
class ActionRef:
    """A single parsed ``uses:`` line."""

    file: Path
    line_index: int
    raw: str
    repo: str
    path_suffix: str
    ref_type: str       # "tag", "branch", or "sha"
    current_ref: str    # value after @
    major: Optional[int] = None

    @property
    def display(self) -> str:
        s = self.repo
        if self.path_suffix:
            s = f"{s}/{self.path_suffix}"
        return s


def parse_uses(line: str, file: Path, line_index: int) -> Optional[ActionRef]:
    m = re.search(r'uses:\s*["\']?([^"\'#\s]+)', line)
    if not m:
        return None
    value = m.group(1)
    if value.startswith("./"):
        return None
    if "@" not in value:
        return None
    prefix, ref = value.rsplit("@", 1)

    parts = prefix.split("/", 2)
    if len(parts) == 2:
        repo, path_suffix = f"{parts[0]}/{parts[1]}", ""
    elif len(parts) == 3 and prefix.count("/") > 1:
        repo, path_suffix = f"{parts[0]}/{parts[1]}", parts[2]
    else:
        return None

    if re.match(r"^[0-9a-f]{40}$", ref):
        return ActionRef(file, line_index, value, repo, path_suffix, "sha", ref)

    if ref.startswith("v") and ref[1:].replace(".", "").isdigit():
        major = int(ref[1:].split(".")[0]) if "." in ref else int(ref[1:])
        return ActionRef(file, line_index, value, repo, path_suffix, "tag", ref, major)

    return ActionRef(file, line_index, value, repo, path_suffix, "branch", ref)


def collect_refs(root: Path) -> List[ActionRef]:
    refs: List[ActionRef] = []
    for dirpath in [root / ".github" / "workflows", root / ".github" / "actions"]:
        if not dirpath.is_dir():
            continue
        for f in sorted(dirpath.rglob("*.yml")):
            if not f.is_file():
                continue
            for i, line in enumerate(f.read_text().splitlines()):
                ar = parse_uses(line, f, i)
                if ar is not None:
                    refs.append(ar)
    return refs


# ---------------------------------------------------------------------------
# GitHub API (authenticated)
# ---------------------------------------------------------------------------

GH_API = "https://api.github.com"


def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_gh_headers())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                if remaining == "0":
                    reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                    wait = max(reset - time.time() + 5, 0)
                    print(
                        f"  rate limit exhausted, sleeping {wait:.0f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            body = exc.read().decode()[:500]
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"URL error: {exc}") from exc
    raise RuntimeError("max retries exceeded")


# Simple in-memory cache for resolved SHAs.
_sha_cache: Dict[str, str] = {}


def resolve_tag_sha(repo: str, tag: str) -> str:
    key = f"{repo}@tag:{tag}"
    if key in _sha_cache:
        return _sha_cache[key]
    ref_data = _gh_get(f"{GH_API}/repos/{repo}/git/ref/tags/{tag}")
    obj = ref_data["object"]
    if obj["type"] == "commit":
        _sha_cache[key] = obj["sha"]
        return obj["sha"]
    tag_data = _gh_get(f"{GH_API}/repos/{repo}/git/tags/{obj['sha']}")
    _sha_cache[key] = tag_data["object"]["sha"]
    return tag_data["object"]["sha"]


def resolve_branch_sha(repo: str, branch: str) -> str:
    key = f"{repo}@branch:{branch}"
    if key in _sha_cache:
        return _sha_cache[key]
    ref_data = _gh_get(f"{GH_API}/repos/{repo}/git/ref/heads/{branch}")
    _sha_cache[key] = ref_data["object"]["sha"]
    return ref_data["object"]["sha"]


def latest_release(repo: str) -> Tuple[str, str]:
    """Return (tag_name, commit_sha) for the latest public release."""
    try:
        release = _gh_get(f"{GH_API}/repos/{repo}/releases/latest")
        tag = release["tag_name"]
    except RuntimeError:
        releases = _gh_get(f"{GH_API}/repos/{repo}/releases?per_page=1")
        if not releases:
            raise RuntimeError(f"No releases found")
        tag = releases[0]["tag_name"]
    sha = resolve_tag_sha(repo, tag)
    return tag, sha


# ---------------------------------------------------------------------------
# Per-ref logic
# ---------------------------------------------------------------------------

@dataclass
class PinResult:
    ref: ActionRef
    status: str            # "uptodate", "updated", "major-bump", "error"
    new_sha: Optional[str] = None
    tag: Optional[str] = None   # display tag for the comment, e.g. "v7.0.0"
    message: str = ""


def _updated(ar: ActionRef, sha: str, tag: str) -> PinResult:
    return PinResult(
        ar, "updated", new_sha=sha, tag=tag,
        message=f"{tag} ({sha[:12]})",
    )


def _resolve_tag(ar: ActionRef) -> Tuple[str, str]:
    """Return (display_tag, commit_sha) for *ar*'s repo at its current best ref."""
    # Branch-only repo — display just the branch name (the repo is in the uses:).
    if ar.repo in BRANCH_ONLY_REPOS:
        for branch in BRANCH_ONLY_REPOS[ar.repo]:
            sha = resolve_branch_sha(ar.repo, branch)
            if sha == ar.current_ref:
                return branch, sha
        # No match -> default to first branch.
        default = BRANCH_ONLY_REPOS[ar.repo][0]
        sha = resolve_branch_sha(ar.repo, default)
        return default, sha

    # Override ref (e.g. codeql-action @v3).
    if ar.repo in REF_OVERRIDES:
        ref = REF_OVERRIDES[ar.repo]
        sha = resolve_tag_sha(ar.repo, ref)
        return ref, sha

    # Normal versioned action.
    tag, sha = latest_release(ar.repo)
    return tag, sha





def check_ref(ar: ActionRef) -> PinResult:
    try:
        tag, sha = _resolve_tag(ar)

        if ar.ref_type == "branch":
            if sha == ar.current_ref:
                return PinResult(ar, "uptodate", tag=tag, message=f"already at {tag}")
            return _updated(ar, sha, tag)

        if ar.ref_type == "sha":
            if sha == ar.current_ref:
                return PinResult(ar, "uptodate", tag=tag, message=f"already at {tag}")
            latest_major = _major_of(tag)
            if latest_major is not None and ar.major is not None and latest_major > ar.major:
                return PinResult(
                    ar, "major-bump",
                    message=f"major bump: v{ar.major} -> v{latest_major} ({tag})",
                )
            return _updated(ar, sha, tag)

        # Tag ref.
        if sha == ar.current_ref:
            return PinResult(ar, "uptodate", tag=tag, message=f"already at {tag}")
        if ar.major is not None:
            latest_major = _major_of(tag)
            if latest_major is not None and latest_major > ar.major:
                return PinResult(
                    ar, "major-bump",
                    message=f"major bump: v{ar.major} -> v{latest_major} ({tag})",
                )
        return _updated(ar, sha, tag)

    except RuntimeError as exc:
        return PinResult(ar, "error", message=str(exc))


def _major_of(tag: str) -> Optional[int]:
    m = re.match(r"^v?(\d+)", tag)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# File rewriting
# ---------------------------------------------------------------------------

_RE_USES = re.compile(r'(uses:\s*["\']?[^"\'#\s]+)@([^\s"\'#]+)')
_RE_TRAILING = re.compile(r"\s+#\s*\S.*$")  # strip existing trailing comment


def _uses_line(prefix: str, sha: str, tag: Optional[str]) -> str:
    """Build a ``uses: owner/repo@sha  # tag`` line."""
    line = f"uses: {prefix}@{sha}"
    if tag:
        line = f"{line}  # {tag}"
    return line


def apply_pin(file: Path, ar: ActionRef, new_sha: str, tag: Optional[str] = None) -> bool:
    lines = file.read_text().splitlines()
    old_line = lines[ar.line_index]
    # Strip any existing trailing comment before rewriting.
    bare = _RE_TRAILING.sub("", old_line).rstrip()
    prefix = ar.repo
    if ar.path_suffix:
        prefix = f"{prefix}/{ar.path_suffix}"
    target = _uses_line(prefix, new_sha, tag)
    new_line = _RE_USES.sub(target, bare)
    if new_line == old_line:
        return False
    lines[ar.line_index] = new_line
    file.write_text("\n".join(lines) + "\n")
    return True


def fix_comment(file: Path, ar: ActionRef, tag: str) -> bool:
    """Ensure the existing SHA-pinned line has the correct tag comment."""
    lines = file.read_text().splitlines()
    old_line = lines[ar.line_index]
    # Strip any existing trailing comment before re-writing.
    bare = _RE_TRAILING.sub("", old_line).rstrip()
    prefix = ar.repo
    if ar.path_suffix:
        prefix = f"{prefix}/{ar.path_suffix}"
    target = _uses_line(prefix, ar.current_ref, tag)
    new_line = _RE_USES.sub(target, bare)
    if new_line == old_line:
        return False
    lines[ar.line_index] = new_line
    file.write_text("\n".join(lines) + "\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Update SHA pins for GitHub Actions",
        epilog="Set GITHUB_TOKEN or GH_TOKEN env var for authenticated API access.",
    )
    ap.add_argument("--write", action="store_true", help="write changes")
    ap.add_argument("--apply-major", action="store_true", help="apply major bumps")
    args = ap.parse_args()

    if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
        print(
            "warning: no GITHUB_TOKEN / GH_TOKEN set — unauthenticated API is "
            "limited to 60 req/hr",
            file=sys.stderr,
        )

    refs = collect_refs(REPO_ROOT)
    if not refs:
        print("No external action references found.")
        return 0

    results: List[PinResult] = []
    for ar in refs:
        results.append(check_ref(ar))

    changed = 0
    comments_fixed = 0
    major_bumps = 0
    errors = 0
    for r in results:
        tag = r.ref.display

        if r.status == "uptodate":
            print(f"  OK  {tag:50s}  {r.message}")
            if args.write and r.tag:
                if fix_comment(r.ref.file, r.ref, r.tag):
                    comments_fixed += 1
                    print(f"       -> comment: {r.ref.file.name}:{r.ref.line_index + 1}")
        elif r.status == "major-bump":
            major_bumps += 1
            note = " (use --apply-major)" if not args.apply_major else ""
            print(f" MAJR {tag:50s}  {r.message}{note}")
            if args.apply_major and r.new_sha:
                if apply_pin(r.ref.file, r.ref, r.new_sha, tag=r.tag):
                    changed += 1
                    print(f"       -> {r.ref.file.name}:{r.ref.line_index + 1}")
        elif r.status == "updated":
            changed += 1
            print(f"  UP  {tag:50s}  {r.message}")
            if args.write and r.new_sha:
                if apply_pin(r.ref.file, r.ref, r.new_sha, tag=r.tag):
                    changed += 1
                    print(f"       -> {r.ref.file.name}:{r.ref.line_index + 1}")
        elif r.status == "error":
            errors += 1
            print(f" ERR  {tag:50s}  {r.message}", file=sys.stderr)

    if errors:
        print(f"\n{errors} error(s)", file=sys.stderr)

    if not args.write and (changed or comments_fixed):
        print(f"\n{changed} update(s) available. Run with --write to apply.")
        return 1

    if changed:
        print(f"\n{changed} file(s) updated.")
    if comments_fixed:
        print(f"{comments_fixed} comment(s) fixed.")

    if not changed and not comments_fixed and major_bumps and not args.apply_major:
        print(f"\n{major_bumps} major-bump(s) available. Review manually or use --apply-major.")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
