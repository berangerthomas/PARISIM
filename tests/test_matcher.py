"""Matcher: method selection, preprocessing, validation and error handling."""

import json

import pytest

from parisim import Matcher, normalize
from parisim.methods import BUILTINS


def _boom(a: str, b: str) -> float:
    raise RuntimeError("boom")


def _with_hook(one_to_many):
    """A method scoring 0.5 pair by pair, with a one_to_many hook."""

    def method(a, b):
        return 0.5

    method.one_to_many = one_to_many
    return method


# -- Selection --------------------------------------------------------------------


def test_default_runs_every_built_in_method():
    assert Matcher().methods == list(BUILTINS)


def test_a_single_name_is_accepted():
    assert Matcher(methods="levenshtein").methods == ["levenshtein"]
    assert Matcher(categories="phonetic").methods == ["soundex", "metaphone"]


def test_empty_selections_select_nothing():
    assert Matcher(methods=[]).methods == []
    assert Matcher(categories=[]).methods == []


def test_methods_take_precedence_over_categories():
    assert Matcher(methods=["levenshtein"], categories=["phonetic"]).methods == [
        "levenshtein"
    ]


def test_unknown_method_suggests_the_closest_name():
    with pytest.raises(ValueError, match="Did you mean 'levenshtein'"):
        Matcher(methods=["levenstein"])


def test_unknown_category_lists_the_available_ones():
    with pytest.raises(ValueError, match=r"Unknown category 'nope'.*'character'"):
        Matcher(categories=["nope"])


def test_available_methods_and_categories():
    assert Matcher.available_methods() == list(BUILTINS)
    assert Matcher.available_categories() == [
        "character",
        "token",
        "phonetic",
    ]


# -- Preprocessing ----------------------------------------------------------------


def test_preprocess_accepts_a_single_function():
    assert (
        Matcher("levenshtein", preprocess=str.lower).compare("ABC", "abc")["levenshtein"]
        == 1.0
    )


def test_preprocess_functions_run_in_order():
    m = Matcher("levenshtein", preprocess=[str.strip, str.lower])
    assert m.compare("  ABC ", "abc")["levenshtein"] == 1.0


def test_results_keep_the_original_strings():
    result = Matcher("levenshtein", preprocess=str.lower).compare("ABC", "abc")
    assert (result.str1, result.str2) == ("ABC", "abc")


def test_preprocess_must_be_functions_returning_strings():
    with pytest.raises(TypeError, match="expects functions"):
        Matcher(preprocess=["lower"])
    with pytest.raises(TypeError, match="returned NoneType"):
        Matcher("levenshtein", preprocess=lambda s: None).compare("a", "b")


# -- Input validation -------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_values_raise_a_clear_error(missing):
    with pytest.raises(TypeError, match="missing value"):
        Matcher().compare(missing, "abc")


def test_non_string_values_raise():
    with pytest.raises(TypeError, match="expected str, got int"):
        Matcher().compare(123, "123")


def test_batch_errors_give_the_position_of_the_bad_pair():
    with pytest.raises(TypeError, match="pair 1: expected str, got float"):
        Matcher().compare_batch([("a", "b"), ("c", float("nan"))])


@pytest.mark.parametrize("bad", ["ab", ("a", "b", "c"), 42])
def test_batch_items_must_be_pairs(bad):
    with pytest.raises(TypeError, match="pair 0: expected a pair of strings"):
        Matcher().compare_batch([bad])


# -- Scores -----------------------------------------------------------------------


def test_every_method_scores_a_typical_pair_without_error():
    result = Matcher().compare("voiture rouge", "automobile rouge")
    assert result.errors == {}
    assert set(result.scores) == set(Matcher().methods) - {"hamming"}


def test_none_means_not_applicable():
    result = Matcher(methods="hamming").compare("a", "abcdef")
    assert result.scores == {}
    assert result.errors == {}


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (85, "ValueError: expected a similarity in [0, 1], got 85"),
        (-0.3, "ValueError"),
        (float("nan"), "ValueError"),
        ("0.5", "TypeError: expected a number, got str"),
    ],
)
def test_invalid_scores_are_rejected(value, error):
    m = Matcher(methods=[])
    m.add_method("bad", lambda a, b: value)
    with pytest.warns(RuntimeWarning, match="'bad' failed"):
        result = m.compare("a", "b")
    assert result.scores == {}
    assert result.errors["bad"].startswith(error)


def test_scores_are_converted_to_plain_floats():
    np = pytest.importorskip("numpy")
    m = Matcher(methods=[])
    m.add_method("numpy", lambda a, b: np.float32(0.25))
    m.add_method("overshoot", lambda a, b: 1 + 1e-12)
    m.add_method("boolean", lambda a, b: a == b)
    scores = m.compare("a", "a").scores
    assert scores == {"numpy": 0.25, "overshoot": 1.0, "boolean": 1.0}
    assert all(type(s) is float for s in scores.values())
    json.dumps(scores)


# -- Errors -----------------------------------------------------------------------


def test_a_failing_method_is_recorded_and_the_others_still_run():
    m = Matcher(methods="levenshtein")
    m.add_method("boom", _boom)
    with pytest.warns(RuntimeWarning, match="'boom' failed on 1 of 1 pair"):
        result = m.compare("a", "b")
    assert result.errors == {"boom": "RuntimeError: boom"}
    assert "levenshtein" in result


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.compare("a", "b"),
        lambda m: m.compare_batch([("a", "b")]),
        lambda m: m.compare_one_to_many("a", ["b"]),
    ],
    ids=["compare", "compare_batch", "compare_one_to_many"],
)
def test_the_failure_warning_points_to_the_callers_line(call):
    m = Matcher(methods=[])
    m.add_method("boom", _boom)
    with pytest.warns(RuntimeWarning) as caught:
        call(m)
    assert caught[0].filename == __file__


def test_strict_mode_raises():
    m = Matcher(methods=[], strict=True)
    m.add_method("boom", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        m.compare("a", "b")


# -- Custom methods ---------------------------------------------------------------


def test_add_method_category():
    m = Matcher(methods=[])
    m.add_method("plain", lambda a, b: 1.0)
    m.add_method("labelled", lambda a, b: 1.0, category="mine")
    assert m.compare("a", "b").categories == {"plain": "custom", "labelled": "mine"}


def test_add_method_replaces_a_method_of_the_same_name():
    m = Matcher(methods="levenshtein")
    m.add_method("levenshtein", lambda a, b: 0.5)
    assert m.compare("a", "a")["levenshtein"] == 0.5
    assert Matcher(methods="levenshtein").compare("a", "a")["levenshtein"] == 1.0


def test_add_method_requires_a_callable():
    with pytest.raises(TypeError, match="must be callable"):
        Matcher().add_method("x", 0.5)


# -- Batches ----------------------------------------------------------------------


def test_compare_batch_accepts_any_iterable_of_pairs():
    batch = Matcher("levenshtein").compare_batch(("a", "a") for _ in range(3))
    assert len(batch) == 3


def test_compare_one_to_many():
    batch = Matcher("levenshtein").compare_one_to_many("paris", ["paris", "pari", "lyon"])
    assert [r.str1 for r in batch] == ["paris"] * 3
    assert [r["levenshtein"] for r in batch] == [1.0, 0.8, 0.0]


def test_compare_one_to_many_gives_the_compare_batch_results():
    m = Matcher(preprocess=normalize)
    query = "Jean Dupont"
    candidates = ["jean dupont", "DUPONT Jean", "Jean-Pierre Dupond", "", "route 66"]
    fast = m.compare_one_to_many(query, candidates)
    slow = m.compare_batch((query, c) for c in candidates)
    assert [r.to_dict() for r in fast] == [r.to_dict() for r in slow]


def test_compare_one_to_many_preprocesses_the_query_once():
    seen = []

    def spy(s):
        seen.append(s)
        return s

    Matcher("levenshtein", preprocess=spy).compare_one_to_many("q", ["a", "b"])
    assert seen == ["q", "a", "b"]


def test_compare_one_to_many_of_no_candidates():
    assert len(Matcher().compare_one_to_many("a", [])) == 0


def test_compare_one_to_many_calls_the_hook_once():
    calls = []

    def one_to_many(query, candidates):
        calls.append((query, candidates))
        return [1.0] * len(candidates)

    m = Matcher(methods=[], preprocess=str.lower)
    m.add_method("hooked", _with_hook(one_to_many))
    assert [r["hooked"] for r in m.compare_one_to_many("Q", ["A", "B"])] == [1.0, 1.0]
    assert calls == [("q", ["a", "b"])]
    assert m.compare("Q", "A")["hooked"] == 0.5  # other calls go pair by pair


@pytest.mark.parametrize(
    ("one_to_many", "errors"),
    [
        (_boom, ["RuntimeError: boom"] * 2),
        (
            lambda q, cs: [1.0],
            ["ValueError: one_to_many: expected 2 scores, one per candidate, got 1"] * 2,
        ),
        (
            lambda q, cs: [1.0, 85],
            [None, "ValueError: expected a similarity in [0, 1], got 85"],
        ),
    ],
    ids=["raises", "wrong-length", "invalid-score"],
)
def test_hook_failures_are_recorded(one_to_many, errors):
    m = Matcher(methods=[])
    m.add_method("hooked", _with_hook(one_to_many))
    with pytest.warns(RuntimeWarning, match="'hooked' failed"):
        batch = m.compare_one_to_many("q", ["a", "b"])
    assert [r.errors.get("hooked") for r in batch] == errors


def test_strict_mode_raises_hook_failures():
    m = Matcher(methods=[], strict=True)
    m.add_method("hooked", _with_hook(_boom))
    with pytest.raises(RuntimeError, match="boom"):
        m.compare_one_to_many("q", ["a"])


def test_len_and_repr():
    m = Matcher(["levenshtein", "soundex"], preprocess=str.lower)
    assert len(m) == 2
    assert repr(m) == "Matcher(methods=['levenshtein', 'soundex'], preprocess=[lower])"
