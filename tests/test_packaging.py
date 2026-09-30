"""Guardrails on what the distribution promises."""

import re
import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

import parisim

DISTRIBUTION = "parisim"


def test_version_is_semantic():
    assert re.fullmatch(r"\d+\.\d+\.\d+", parisim.__version__)


def test_public_api_resolves_and_is_documented():
    for name in parisim.__all__:
        obj = getattr(parisim, name)
        assert name == "__version__" or obj.__doc__, f"{name} has no docstring"


def test_py_typed_marker_ships_with_the_package():
    assert (Path(parisim.__file__).parent / "py.typed").is_file()


def test_core_dependencies_stay_light():
    """Everything heavy (numpy, pandas, matplotlib, models) belongs in extras."""
    core = [r for r in requires(DISTRIBUTION) or [] if "extra ==" not in r]
    assert {re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0].lower() for r in core} == {
        "jellyfish",
        "rapidfuzz",
    }


def test_importing_parisim_does_not_import_optional_stacks():
    code = (
        "import sys, parisim\n"
        "parisim.Matcher().compare('a', 'b')\n"
        "optional = ('fastembed', 'matplotlib', 'numpy', 'pandas', 'sklearn')\n"
        "print(sorted(m for m in optional if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
