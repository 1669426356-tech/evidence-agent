"""Check and export only this standalone source tree; never copy Git history.

This is a focused preflight, not a comprehensive credential/security scanner.
Use --candidate for an explicitly unlicensed LOCAL review archive. Public
exports require a LICENSE and project.license metadata selected by the author.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {
    "README.md", "LICENSE", "pyproject.toml", "CONTRIBUTING.md",
    "SECURITY.md", "CHANGELOG.md", ".gitignore", ".gitattributes", ".env.example",
    "MANIFEST.in", "constraints-tested.txt",
    "THIRD_PARTY_NOTICES.md",
}
SOURCE_DIRS = {"src", "tests", "docs", "examples", "scripts", ".github"}
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", ".state", "outputs"}
EXTENSIONS = {".py", ".md", ".toml", ".yml", ".yaml", ".typed"}
PATTERNS = {
    "credential-shaped literal": re.compile(r"\b(?:sk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "author-local path": re.compile(r"(?i)[A-Z]:[\\/](?:Users|codex|agent)[\\/]"),
}


def inspect_tree(root: Path = ROOT) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    problems: list[str] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        retained = []
        for name in sorted(names):
            candidate = parent / name
            if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
                problems.append(f"linked directory excluded: {candidate.relative_to(root).as_posix()}")
            elif name not in EXCLUDED_DIRS and not name.endswith(".egg-info"):
                retained.append(name)
        names[:] = retained
        for name in sorted(filenames):
            path = parent / name
            relative = path.relative_to(root)
            label = relative.as_posix()
            if path.is_symlink():
                problems.append(f"linked file excluded: {label}")
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if len(relative.parts) == 1 and name == "SOURCE_MANIFEST.json":
                # Exported snapshots carry a generated manifest; regenerate it
                # instead of embedding a stale manifest or a duplicate ZIP entry.
                continue
            if ((len(relative.parts) == 1 and name in TOP_FILES)
                    or (relative.parts[0] in SOURCE_DIRS and path.suffix in EXTENSIONS)):
                files.append(path)
            else:
                problems.append(f"unexpected file excluded: {label}")
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                problems.append(f"non-text or unreadable source: {label}")
                continue
            for reason, pattern in PATTERNS.items():
                for match in pattern.finditer(content):
                    line = content.count("\n", 0, match.start()) + 1
                    # Never print the matched content.
                    problems.append(f"{reason}: {label}:{line}")
    return sorted(files), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Export a ZIP outside the source directory")
    parser.add_argument("--candidate", action="store_true", help="Allow a local candidate with licensing pending")
    args = parser.parse_args()
    files, problems = inspect_tree()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    repository_name = metadata.get("tool", {}).get("agent_framework", {}).get("release", {}).get("repository-name", project["name"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", repository_name):
        problems.append("Invalid repository name in release configuration")
    licensed = (ROOT / "LICENSE").is_file() and bool(project.get("license"))
    if not licensed and not args.candidate:
        problems.append("License selection pending: add LICENSE and project.license before public export")
    if problems:
        print(json.dumps({"passed": False, "problems": problems}, ensure_ascii=False, indent=2))
        return 1
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = {"passed": True, "repository_name": repository_name, "file_count": len(files), "license_selected": licensed,
              "candidate_only": not licensed, "git_history_included": False}
    if args.archive:
        target = args.archive.resolve()
        if target.is_relative_to(ROOT):
            parser.error("Archive must be outside the source tree")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Create-only prevents quietly replacing a previously reviewed archive.
        prefix = f"{repository_name}-{project['version']}"
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, prefix + "/" + path.relative_to(ROOT).as_posix())
            archive.writestr(prefix + "/SOURCE_MANIFEST.json", json.dumps(
                {"scope": "local candidate" if not licensed else "source release", "sha256": manifest},
                sort_keys=True, indent=2,
            ) + "\n")
        report["archive"] = target.name
        report["archive_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
