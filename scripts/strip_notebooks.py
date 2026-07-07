#!/usr/bin/env python3
"""Git clean filter: strips execution artifacts from a notebook piped on stdin."""
import json
import sys


def strip_notebook(data: dict) -> None:
    for cell in data.get("cells", []):
        if cell.get("outputs"):
            cell["outputs"] = []
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
    metadata = data.get("metadata", {})
    metadata.get("kernelspec", {}).pop("display_name", None)
    metadata.get("language_info", {}).pop("version", None)


def main() -> int:
    data = json.load(sys.stdin)
    strip_notebook(data)
    json.dump(data, sys.stdout, indent=1, ensure_ascii=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
