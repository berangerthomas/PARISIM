"""The example notebook runs top to bottom against the current API."""

import json
import warnings
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "demo.ipynb"


def test_example_notebook_runs():
    """Execute every code cell, except those tagged "needs-model" (download)."""
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [
        (i, "".join(cell["source"]))
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        and "needs-model" not in cell["metadata"].get("tags", [])
    ]
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the notebook shows one
        for i, source in code_cells:
            exec(compile(source, f"demo.ipynb, cell {i}", "exec"), namespace)
    plt.close("all")
