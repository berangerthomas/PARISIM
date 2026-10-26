"""Built-in similarity methods.

Each method is a plain function ``(a, b) -> float | None``: a similarity in
[0, 1], 1 meaning identical, or ``None`` when the method does not apply to the
pair. Use them through a ``Matcher``, or on their own:

>>> from parisim.methods import jaro_winkler, token_sort
>>> round(jaro_winkler("martha", "marhta"), 3)
0.961
>>> token_sort("red car", "car red")
1.0
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeVar

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import Hamming, Indel, JaroWinkler, Levenshtein

__all__ = [
    "BUILTINS",
    "Similarity",
    "hamming",
    "indel_ratio",
    "jaro_winkler",
    "levenshtein",
    "metaphone",
    "partial_ratio",
    "soundex",
    "tfidf_cosine",
    "token_set",
    "token_sort",
]

#: A similarity method: ``(a, b) -> score in [0, 1]``, or ``None`` if not applicable.
Similarity = Callable[[str, str], float | None]

_Method = TypeVar("_Method", bound=Similarity)
_Repr = TypeVar("_Repr")


def _one_to_many(
    represent: Callable[[str], _Repr], compare: Callable[[_Repr, _Repr], float]
) -> Callable[[_Method], _Method]:
    """Decorate a method computing ``compare(represent(a), represent(b))``.

    The method gets a ``one_to_many(query, candidates)`` attribute, which
    ``Matcher.compare_one_to_many`` calls instead of the method: it represents
    the query once, not once per candidate.
    """

    def one_to_many(query: str, candidates: Sequence[str]) -> list[float]:
        represented = represent(query)
        return [compare(represented, represent(c)) for c in candidates]

    def decorate(method: _Method) -> _Method:
        method.one_to_many = one_to_many  # type: ignore[attr-defined]
        return method

    return decorate


# -- Character level ------------------------------------------------------------


def levenshtein(a: str, b: str) -> float:
    """``1 - edits / max length``, where an insertion, deletion or substitution
    counts as one edit. The reference measure for typos."""
    return Levenshtein.normalized_similarity(a, b)


def indel_ratio(a: str, b: str) -> float:
    """The classic ``fuzz.ratio``: like Levenshtein, but a substitution counts as
    two edits (a deletion plus an insertion)."""
    return Indel.normalized_similarity(a, b)


def hamming(a: str, b: str) -> float | None:
    """Share of positions holding the same character. Only defined for strings
    of equal length (codes, identifiers, postcodes), ``None`` otherwise."""
    if len(a) != len(b):
        return None
    return Hamming.normalized_similarity(a, b)


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler: tolerant to transpositions, rewards a common prefix. Suited
    to short strings such as names."""
    return JaroWinkler.normalized_similarity(a, b)


def partial_ratio(a: str, b: str) -> float:
    """Best alignment of the shorter string within the longer one: 1.0 when one
    is a substring of the other, e.g. an abbreviation or a truncated field."""
    return fuzz.partial_ratio(a, b) / 100


# -- Word level -----------------------------------------------------------------


def token_sort(a: str, b: str) -> float:
    """``fuzz.ratio`` after sorting the words: ignores word order."""
    return fuzz.token_sort_ratio(a, b) / 100


def token_set(a: str, b: str) -> float:
    """Compares the words both strings share with the words only one has:
    ignores word order, repeated words and extra words."""
    if not a.split() and not b.split():  # rapidfuzz scores two word-less strings 0
        return 1.0
    return fuzz.token_set_ratio(a, b) / 100


# scikit-learn's default token pattern: words of two or more characters.
_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
# Smoothed idf of a word found in one of the two documents,
# ln((1 + n) / (1 + df)) + 1 with n = 2, df = 1. A word found in both gets 1.
_IDF_UNSHARED = math.log(3 / 2) + 1

_Words = tuple[str, Counter[str]]


def _words(s: str, token_pattern: str = _TOKEN_PATTERN, lowercase: bool = True) -> _Words:
    """*s*, lowercased if requested, and the count of each of its words."""
    if lowercase:
        s = s.lower()
    return s, Counter(re.findall(token_pattern, s))


def _tfidf(wa: _Words, wb: _Words) -> float:
    """``tfidf_cosine`` of two strings, given as ``_words(a)`` and ``_words(b)``."""
    (a, ta), (b, tb) = wa, wb
    if not ta or not tb:
        return float(a == b)
    dot = sum(count * tb[word] for word, count in ta.items() if word in tb)
    norm_a = sum((c if w in tb else c * _IDF_UNSHARED) ** 2 for w, c in ta.items())
    norm_b = sum((c if w in ta else c * _IDF_UNSHARED) ** 2 for w, c in tb.items())
    return dot / math.sqrt(norm_a * norm_b)


@_one_to_many(_words, _tfidf)
def tfidf_cosine(
    a: str,
    b: str,
    *,
    token_pattern: str = _TOKEN_PATTERN,
    lowercase: bool = True,
) -> float:
    """Cosine similarity of TF-IDF word vectors fitted on the two strings.

    Equal to scikit-learn's ``TfidfVectorizer`` + cosine on ``[a, b]``: words of
    two or more characters, lowercased, smoothed idf. With two documents, idf
    only weighs up the words found in one string, so this measures vocabulary
    overlap. To keep one-letter words, use
    ``functools.partial(tfidf_cosine, token_pattern=r"(?u)\\b\\w+\\b")``.
    """
    return _tfidf(
        _words(a, token_pattern, lowercase), _words(b, token_pattern, lowercase)
    )


# -- Phonetic -------------------------------------------------------------------

_LATIN_LETTER = re.compile("[A-Za-z]")
_Codes = Counter[tuple[str, str]]


def _phonetic_codes(s: str, encode: Callable[[str], str]) -> _Codes:
    """Phonetic code of each word with Latin letters; other words (numbers,
    punctuation, other scripts) are kept as is, so "66" and "67" differ."""
    return Counter(
        ("code", encode(w) or w) if _LATIN_LETTER.search(w) else ("word", w)
        for w in s.split()
    )


def _phonetic_ratio(ca: _Codes, cb: _Codes) -> float:
    """Share of words that sound alike, pairing each word with at most one."""
    if not ca or not cb:
        return float(not ca and not cb)
    return (ca & cb).total() / max(ca.total(), cb.total())


_soundex_codes = partial(_phonetic_codes, encode=jellyfish.soundex)
_metaphone_codes = partial(_phonetic_codes, encode=jellyfish.metaphone)


@_one_to_many(_soundex_codes, _phonetic_ratio)
def soundex(a: str, b: str) -> float:
    """Share of words with the same Soundex code. English-centric; words must
    start with the same letter: Robert ~ Rupert, but Philip !~ Filip."""
    return _phonetic_ratio(_soundex_codes(a), _soundex_codes(b))


@_one_to_many(_metaphone_codes, _phonetic_ratio)
def metaphone(a: str, b: str) -> float:
    """Share of words with the same Metaphone code, from English pronunciation
    rules: Philip ~ Filip, Knight ~ Night, but Robert !~ Rupert."""
    return _phonetic_ratio(_metaphone_codes(a), _metaphone_codes(b))


# -- Catalog --------------------------------------------------------------------

#: Every built-in method, as ``name: (category, function)``.
BUILTINS: dict[str, tuple[str, Similarity]] = {
    "levenshtein": ("character", levenshtein),
    "indel_ratio": ("character", indel_ratio),
    "hamming": ("character", hamming),
    "jaro_winkler": ("character", jaro_winkler),
    "partial_ratio": ("character", partial_ratio),
    "token_sort": ("token", token_sort),
    "token_set": ("token", token_set),
    "tfidf_cosine": ("token", tfidf_cosine),
    "soundex": ("phonetic", soundex),
    "metaphone": ("phonetic", metaphone),
}
