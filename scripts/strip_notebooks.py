#!/usr/bin/env python3
"""Git clean filter: reduces a notebook piped on stdin to its source content.

Everything an editor (Jupyter Notebook, JupyterLab, VS Code, Colab, ...) may
write when opening or running a notebook is dropped or replaced by a fixed
value, so that executing a notebook in any environment never shows up as a
change. Only cell types, sources, attachments and tags are kept.
"""
import json
import sys

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
LANGUAGE_INFO = {"name": "python"}
KEPT_CELL_METADATA = ("tags",)
# Cell ids were introduced in nbformat 4.5; stripping them requires declaring
# 4.4 so that the committed notebook stays schema-valid.
NBFORMAT_MINOR_WITHOUT_IDS = 4


def normalize_source(source: list | str) -> list:
    """Normalize cell source to nbformat's canonical line-list form."""
    if isinstance(source, list):
        source = "".join(source)
    lines = source.splitlines(keepends=True)
    return lines if lines else [""]


def strip_cell(cell: dict) -> None:
    cell.pop("id", None)
    metadata = cell.get("metadata", {})
    cell["metadata"] = {k: metadata[k] for k in KEPT_CELL_METADATA if k in metadata}
    if "source" in cell:
        cell["source"] = normalize_source(cell["source"])
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    else:
        cell.pop("outputs", None)
        cell.pop("execution_count", None)


def strip_notebook(data: dict) -> None:
    for cell in data.get("cells", []):
        strip_cell(cell)
    data["metadata"] = {"kernelspec": dict(KERNELSPEC), "language_info": dict(LANGUAGE_INFO)}
    if data.get("nbformat") == 4:
        data["nbformat_minor"] = min(
            data.get("nbformat_minor", NBFORMAT_MINOR_WITHOUT_IDS), NBFORMAT_MINOR_WITHOUT_IDS
        )


def main() -> int:
    data = json.loads(sys.stdin.buffer.read())
    strip_notebook(data)
    # Sorted keys make the output independent of the key order each editor uses.
    output = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
