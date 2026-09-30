"""MatchResult and BatchResult: display, exports and charts."""

import json
import math
import sys

import pytest

from parisim import BatchResult, Matcher, MatchResult

PAIRS = [
    ("voiture rouge", "automobile rouge"),
    ("chat noir", "chats noirs"),
    ("ab", "ab"),
]


@pytest.fixture
def result():
    return Matcher(["levenshtein", "token_sort"]).compare("red car", "car red")


@pytest.fixture
def batch():
    return Matcher().compare_batch(PAIRS)


# -- MatchResult ------------------------------------------------------------------


def test_repr_lists_scores_from_best_to_worst(result):
    assert repr(result) == (
        "MatchResult('red car' vs 'car red')\n  token_sort   1.000\n  levenshtein  0.143"
    )


def test_repr_shows_failures():
    r = MatchResult("a", "b", scores={"x": 0.5}, errors={"boom": "RuntimeError: boom"})
    assert repr(r).splitlines()[1:] == [
        "  x     0.500",
        "  boom  failed: RuntimeError: boom",
    ]


def test_repr_without_scores():
    assert repr(MatchResult("a", "b")) == "MatchResult('a' vs 'b', no scores)"


def test_repr_truncates_long_strings():
    assert "…" in repr(MatchResult("x" * 100, "y"))


def test_getitem_and_contains(result):
    assert result["token_sort"] == 1.0
    assert "token_sort" in result
    assert "soundex" not in result
    with pytest.raises(KeyError):
        result["soundex"]


def test_to_dict_is_json_serializable(result):
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["scores"]["token_sort"] == 1.0
    assert payload["categories"] == {"levenshtein": "character", "token_sort": "token"}


# -- BatchResult ------------------------------------------------------------------


def test_batch_protocol(batch):
    assert len(batch) == 3
    assert isinstance(batch[0], MatchResult)
    assert [r.str2 for r in batch] == [b for _, b in PAIRS]
    assert repr(batch) == f"BatchResult(3 pairs, {len(Matcher())} methods)"


def test_batch_repr_counts_pairs_with_errors():
    m = Matcher("levenshtein")
    m.add_method("picky", lambda a, b: 1 / (len(a) - 1))
    with pytest.warns(RuntimeWarning):
        batch = m.compare_batch([("ab", "ab"), ("a", "a")])
    assert repr(batch) == "BatchResult(2 pairs, 2 methods, 1 pairs with errors)"


def test_batch_to_dict_is_json_serializable(batch):
    payload = json.loads(json.dumps(batch.to_dict()))
    assert [p["str1"] for p in payload["pairs"]] == [a for a, _ in PAIRS]


def test_to_pandas_has_one_row_per_pair_and_one_column_per_method(batch):
    pytest.importorskip("pandas")
    df = batch.to_pandas()
    assert list(df.columns) == ["str1", "str2", *Matcher().methods]
    assert df.shape == (3, 2 + len(Matcher()))
    assert df.loc[2, "levenshtein"] == 1.0


def test_to_pandas_uses_nan_where_a_method_gave_no_score(batch):
    pytest.importorskip("pandas")
    hamming = batch.to_pandas()["hamming"]
    assert hamming.dtype == float
    assert math.isnan(hamming[0])  # strings of different lengths
    assert hamming[2] == 1.0


def test_to_pandas_on_an_empty_batch():
    pytest.importorskip("pandas")
    assert list(BatchResult().to_pandas().columns) == ["str1", "str2"]


def test_to_pandas_explains_how_to_install_pandas(monkeypatch, batch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="pip install pandas"):
        batch.to_pandas()


# -- Charts -----------------------------------------------------------------------


@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_result_plot_draws_one_bar_per_score_under_category_headers(plt, result):
    ax = result.plot()
    assert len(ax.patches) == 2
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["", "levenshtein", "", "token_sort"]  # a header row per category
    assert {t.get_text() for t in ax.texts} >= {"character", "token", "0.14", "1.00"}


def test_batch_plot_draws_one_box_per_method(plt, batch):
    ax = batch.plot()
    labels = {t.get_text() for t in ax.get_yticklabels()} - {""}
    assert labels == set(Matcher().methods)


def test_plots_draw_on_the_given_axes(plt, result, batch):
    _, (left, right) = plt.subplots(1, 2)
    assert result.plot(ax=left) is left
    assert batch.plot(ax=right) is right


def test_plotting_nothing_says_so(plt):
    ax = MatchResult("a", "b").plot()
    assert [t.get_text() for t in ax.texts] == ["No scores to plot"]


def test_plot_explains_how_to_install_matplotlib(monkeypatch, result):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="pip install"):
        result.plot()
