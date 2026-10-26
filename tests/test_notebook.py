"""The example notebook runs top to bottom against the current API."""

import json
import sys
import types
import warnings
from pathlib import Path

import pytest

from parisim.embeddings import _load

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "demo.ipynb"


class VowelCounts:
    """Stands in for fastembed's TextEmbedding: no model to download."""

    def __init__(self, model_name):
        pass

    def embed(self, texts):
        return ([text.count(vowel) for vowel in "aeiou"] for text in texts)


@pytest.fixture
def fake_model(monkeypatch):
    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = VowelCounts
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    _load.cache_clear()
    yield
    _load.cache_clear()


def test_example_notebook_runs(fake_model):
    """Execute every code cell, the embedding model replaced by a fake."""
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [
        (i, "".join(cell["source"]))
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code"
    ]
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        # The notebook shows this one; any other method failure fails the test.
        warnings.filterwarnings("ignore", "method 'fragile' failed", RuntimeWarning)
        for i, source in code_cells:
            exec(compile(source, f"demo.ipynb, cell {i}", "exec"), namespace)
    plt.close("all")
