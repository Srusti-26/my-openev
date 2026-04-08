#!/usr/bin/env python3
"""
Standalone validator — works with any openenv-core version.
Mimics openenv validate output exactly.
"""
import sys
from pathlib import Path

def validate(env_path: Path):
    issues = []

    if not (env_path / "pyproject.toml").exists():
        issues.append("Missing pyproject.toml")
        return False, issues

    if not (env_path / "uv.lock").exists():
        issues.append("Missing uv.lock")

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(env_path / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
    except Exception as e:
        issues.append(f"Failed to parse pyproject.toml: {e}")
        return False, issues

    scripts = pyproject.get("project", {}).get("scripts", {})
    if "server" not in scripts:
        issues.append("Missing [project.scripts] server entry point")

    server_entry = scripts.get("server", "")
    if server_entry and ":main" not in server_entry:
        issues.append(f"Server entry point should reference main function, got: {server_entry}")

    deps = [d.lower() for d in pyproject.get("project", {}).get("dependencies", [])]
    has_openenv = any(d.startswith("openenv") and not d.startswith("openenv-core") for d in deps)
    has_core = any(d.startswith("openenv-core") for d in deps)
    if not (has_openenv or has_core):
        issues.append("Missing required dependency: openenv>=0.2.0")

    server_app = env_path / "server" / "app.py"
    if not server_app.exists():
        issues.append("Missing server/app.py")
    else:
        content = server_app.read_text(encoding="utf-8")
        if "def main(" not in content:
            issues.append("server/app.py missing main() function")
        if "__name__" not in content or "main()" not in content:
            issues.append("server/app.py main() function not callable")

    return len(issues) == 0, issues


if __name__ == "__main__":
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    name = env_path.name
    ok, issues = validate(env_path)
    if ok:
        print(f"[OK] {name}: Ready for multi-mode deployment")
        sys.exit(0)
    else:
        print(f"[FAIL] {name}: Not ready for multi-mode deployment\n")
        print("Issues found:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
