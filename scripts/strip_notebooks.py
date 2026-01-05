#!/usr/bin/env python3
import json
from pathlib import Path


def strip_notebook(path: Path) -> bool:
    data = json.loads(path.read_text())
    dirty = False
    for cell in data.get("cells", []):
        if cell.get("outputs"):
            cell["outputs"] = []
            dirty = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            dirty = True
    if dirty:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=True) + "\n")
    return dirty


def main() -> int:
    changed = []
    for nb in Path("notebooks").rglob("*.ipynb"):
        if strip_notebook(nb):
            changed.append(nb)
    if changed:
        print("Stripped outputs from:")
        for nb in changed:
            print(f"  {nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
