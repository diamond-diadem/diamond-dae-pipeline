import pathlib


def _find_project_root():
    """Locate the repository root (the directory holding `pyproject.toml` and `notebooks-pipeline/`).

    Looks next to the package first (editable install, the default with `uv sync`),
    then upwards from the current working directory (non-editable install, notebooks
    run from inside the repository). Falls back to the package-relative guess.
    """
    package_guess = pathlib.Path(__file__).resolve().parents[2]
    for start in (package_guess, pathlib.Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "notebooks-pipeline").is_dir():
                return candidate
    return package_guess


PROJECT_ROOT = _find_project_root()
