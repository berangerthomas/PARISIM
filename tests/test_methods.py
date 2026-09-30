"""Built-in similarity methods: shared invariants, then per-method semantics."""

import math
import random
from functools import partial

import pytest

from parisim.methods import (
    BUILTINS,
    OPT_IN,
    hamming,
    indel_ratio,
    jaro_winkler,
    levenshtein,
    metaphone,
    partial_ratio,
    soundex,
    tfidf_cosine,
    token_set,
    token_sort,
)

# Every built-in method that runs without a model.
STRING_METHODS = {name: fn for name, (_, fn) in BUILTINS.items() if name not in OPT_IN}

EDGE_STRINGS = [
    "",
    " ",
    "!",
    "a",
    "a b",
    "42",
    "route 66",
    "東京",
    "😀",
    "Émile",
    "l'église",
    "x" * 300,
]

PAIRS = [
    ("kitten", "sitting"),
    ("voiture rouge", "automobile rouge"),
    ("jean-pierre dupont", "dupont jean pierre"),
    ("the quick brown fox", "quick fox"),
    ("smith schmidt", "schmidt smith"),
    ("route 66", "route 67"),
    ("a", "abc"),
    ("", "x"),
    ("東京", "大阪"),
]


# -- Invariants -----------------------------------------------------------------


@pytest.mark.parametrize("name", STRING_METHODS)
@pytest.mark.parametrize("s", EDGE_STRINGS)
def test_identical_strings_score_one(name, s):
    assert STRING_METHODS[name](s, s) == pytest.approx(1.0)


@pytest.mark.parametrize("name", STRING_METHODS)
@pytest.mark.parametrize(("a", "b"), PAIRS)
def test_methods_are_symmetric(name, a, b):
    fn = STRING_METHODS[name]
    assert fn(a, b) == fn(b, a)


@pytest.mark.parametrize("name", STRING_METHODS)
@pytest.mark.parametrize(("a", "b"), PAIRS)
def test_scores_are_floats_within_unit_range(name, a, b):
    score = STRING_METHODS[name](a, b)
    assert score is None or (isinstance(score, float) and 0.0 <= score <= 1.0)


def test_catalog_categories():
    categories = {name: category for name, (category, _) in BUILTINS.items()}
    assert set(categories.values()) == {"character", "token", "phonetic", "semantic"}
    assert set(BUILTINS) >= OPT_IN
    assert categories["embedding"] == "semantic"


# -- Character level --------------------------------------------------------------


def test_levenshtein_counts_a_substitution_as_one_edit():
    # kitten -> sitting: 3 edits over 7 characters.
    assert levenshtein("kitten", "sitting") == pytest.approx(1 - 3 / 7)


def test_indel_ratio_counts_a_substitution_as_two_edits():
    assert indel_ratio("abcd", "abxd") == pytest.approx(0.75)  # 2 edits / 8 characters
    assert levenshtein("abcd", "abxd") == pytest.approx(0.75)  # 1 edit / 4 characters
    assert indel_ratio("kitten", "sitting") > levenshtein("kitten", "sitting")


def test_hamming_compares_positions_of_equal_length_strings():
    assert hamming("75001", "75002") == pytest.approx(0.8)
    assert hamming("", "") == 1.0


def test_hamming_does_not_apply_to_strings_of_different_lengths():
    assert hamming("a", "abcdef") is None


def test_jaro_winkler_rewards_a_common_prefix():
    assert jaro_winkler("prefix_a", "prefix_b") > jaro_winkler("a_suffix", "b_suffix")


def test_partial_ratio_finds_substrings():
    assert partial_ratio("car", "a red car indeed") == 1.0


# -- Word level -------------------------------------------------------------------


def test_token_sort_ignores_word_order():
    assert token_sort("red car", "car red") == 1.0


def test_token_set_ignores_duplicate_and_extra_words():
    assert token_set("car car red", "red car") == 1.0
    assert token_set("red car", "big red car") == 1.0


def test_tfidf_disjoint_vocabularies_score_zero():
    assert tfidf_cosine("alpha beta", "gamma delta") == 0.0


def test_tfidf_scores_zero_when_one_side_has_no_words():
    assert tfidf_cosine("", "hello world") == 0.0
    assert tfidf_cosine("!", "hello world") == 0.0


def test_tfidf_is_case_insensitive_by_default():
    assert tfidf_cosine("Hello World", "hello world") == pytest.approx(1.0)
    assert tfidf_cosine("Hello", "hello", lowercase=False) == 0.0


def test_tfidf_token_pattern_can_keep_one_letter_words():
    single_letters = partial(tfidf_cosine, token_pattern=r"(?u)\b\w+\b")
    idf = math.log(1.5) + 1  # "b" and "c" are each in one document, "a" in both
    assert single_letters("a b", "a c") == pytest.approx(1 / (1 + idf**2))
    assert tfidf_cosine("a b", "a c") == 0.0  # the default needs 2+ characters


def test_tfidf_matches_the_scikit_learn_reference():
    """The pure-Python TF-IDF must equal sklearn's TfidfVectorizer + cosine."""
    pytest.importorskip("sklearn", reason="reference implementation")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def reference(a: str, b: str) -> float:
        try:
            tfidf = TfidfVectorizer().fit_transform([a, b])
        except ValueError:  # empty vocabulary: neither string has a word
            return float(a.lower() == b.lower())
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])

    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy",
        "dog", "cafe", "voiture", "rouge", "x", "ab", "42",
    ]  # fmt: skip
    rng = random.Random(0)
    cases = [
        ("voiture rouge", "automobile rouge"),
        ("chat noir", "chats noirs"),
        ("paris france", "paris"),
        ("the the the", "the"),
        ("", ""),
        ("a", "b"),
        ("A B C", "c b a"),
        ("東京", "东京"),
    ]
    cases += [
        (
            " ".join(rng.choice(words) for _ in range(rng.randint(0, 8))),
            " ".join(rng.choice(words) for _ in range(rng.randint(0, 8))),
        )
        for _ in range(200)
    ]
    for a, b in cases:
        assert tfidf_cosine(a, b) == pytest.approx(reference(a, b), abs=1e-12), (a, b)


# -- Phonetic ---------------------------------------------------------------------


def test_soundex_and_metaphone_complement_each_other():
    assert soundex("Robert", "Rupert") == 1.0
    assert metaphone("Robert", "Rupert") == 0.0
    assert soundex("Philip", "Filip") == 0.0
    assert metaphone("Philip", "Filip") == 1.0


def test_phonetic_methods_compare_word_by_word():
    # Encoding the whole phrase would only keep the first word.
    assert soundex("voiture rouge", "automobile rouge") == pytest.approx(0.5)
    assert metaphone("smith paris", "schmidt paris") == pytest.approx(0.5)


def test_phonetic_partial_credit_scales_with_the_longer_side():
    assert soundex("smith", "smith jones taylor") == pytest.approx(1 / 3)


@pytest.mark.parametrize("method", [soundex, metaphone])
def test_phonetic_methods_require_numbers_to_match_exactly(method):
    # Soundex codes "66" and "67" alike and Metaphone drops digits.
    assert method("route 66", "route 67") == pytest.approx(0.5)
    assert method("12 rue de la paix", "12 rue de la paix") == 1.0


@pytest.mark.parametrize("method", [soundex, metaphone])
def test_phonetic_methods_compare_other_scripts_exactly(method):
    assert method("東京 tower", "大阪 tower") == pytest.approx(0.5)


@pytest.mark.parametrize("method", [soundex, metaphone])
def test_phonetic_methods_on_word_less_strings(method):
    assert method("", "  ") == 1.0
    assert method("", "robert") == 0.0
