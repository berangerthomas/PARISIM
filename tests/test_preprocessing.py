"""Preprocessing functions."""

import unicodedata

import pytest

from parisim import (
    Matcher,
    normalize,
    normalize_whitespace,
    strip_accents,
    strip_punctuation,
)


def test_normalize_does_the_usual_cleanup():
    assert normalize("  Jean-François   L'HÔPITAL! ") == "jean francois l hopital"
    assert normalize("STRASSE") == normalize("Straße") == "strasse"
    assert normalize("Cœur") == "coeur"


def test_normalize_makes_matcher_ignore_case_accents_and_punctuation():
    m = Matcher("levenshtein", preprocess=normalize)
    assert m.compare("Café-Crème !", "cafe creme")["levenshtein"] == 1.0


# -- strip_accents ----------------------------------------------------------------


def test_strip_accents_handles_precomposed_and_decomposed_forms():
    assert strip_accents("\u00e9") == strip_accents("e\u0301") == "e"


def test_strip_accents_splits_letters_unicode_does_not_decompose():
    assert strip_accents("œ æ ø ł đ Œ Æ Ø Ł Đ") == "oe ae o l d OE AE O L D"


def test_strip_accents_expands_compatibility_characters():
    assert strip_accents("ﬁnance") == "finance"


@pytest.mark.parametrize("text", ["ガス", "한국", "हिन्दी", "ไทย"])
def test_strip_accents_leaves_other_scripts_intact(text):
    # Japanese voicing marks or Devanagari signs are not accents: removing
    # them turns ガス (gas) into カス (dregs).
    assert strip_accents(text) == unicodedata.normalize("NFC", text)


def test_strip_accents_returns_composed_text():
    assert strip_accents("Ångström 한국") == "Angstrom 한국"


# -- strip_punctuation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("l'avion", "l avion"),
        ("jean-pierre", "jean pierre"),
        ("hello_world", "hello world"),
        ("Café!", "Café"),
        ("prix : 12 € (TTC)", "prix 12 TTC"),
        ("3,14 = 3.14", "3 14 3 14"),
    ],
)
def test_strip_punctuation_replaces_punctuation_and_symbols_with_spaces(text, expected):
    assert strip_punctuation(text) == expected


@pytest.mark.parametrize("text", ["re\u0301sume\u0301", "हिन्दी", "ไทย"])
def test_strip_punctuation_keeps_combining_marks(text):
    # A regex such as [^\w\s] treats them as punctuation and splits words.
    assert strip_punctuation(text) == text


def test_normalize_whitespace():
    assert normalize_whitespace("  a \t b\n\nc  ") == "a b c"
