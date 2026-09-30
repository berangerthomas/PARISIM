"""Preprocessing functions for ``Matcher(preprocess=...)``.

Any ``str -> str`` function works, such as ``str.lower`` or ``str.casefold``.
``normalize`` does the usual cleanup in one step.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize", "normalize_whitespace", "strip_accents", "strip_punctuation"]

# Combining diacritical marks of the Latin, Greek and Cyrillic scripts. Other
# scripts' combining marks are not accents (Japanese voicing marks, Devanagari
# vowel signs...) and are left alone.
_DIACRITICS = re.compile("[\u0300-\u036f]")

# Letters Unicode does not decompose into a base letter and an accent.
_LETTERS = str.maketrans(
    {"œ": "oe", "æ": "ae", "ø": "o", "ł": "l", "đ": "d"}
    | {"Œ": "OE", "Æ": "AE", "Ø": "O", "Ł": "L", "Đ": "D"}
)


class _PunctuationToSpace(dict[int, "str | int"]):
    """``str.translate`` table mapping punctuation and symbols (Unicode
    categories P* and S*) to a space, filled in as characters are met."""

    def __missing__(self, codepoint: int) -> str | int:
        category = unicodedata.category(chr(codepoint))
        self[codepoint] = " " if category[0] in "PS" else codepoint
        return self[codepoint]


_PUNCTUATION = _PunctuationToSpace()


def strip_accents(s: str) -> str:
    """Remove accents from Latin letters and split ligatures.

    >>> strip_accents("Œuvre complète à Łódź")
    'OEuvre complete a Lodz'
    """
    decomposed = unicodedata.normalize("NFKD", s.translate(_LETTERS))
    return unicodedata.normalize("NFC", _DIACRITICS.sub("", decomposed))


def strip_punctuation(s: str) -> str:
    """Replace punctuation and symbols with spaces, then collapse whitespace.

    Words joined by punctuation stay separate words.

    >>> strip_punctuation("L'avion de Jean-Pierre, à 12h30 !")
    'L avion de Jean Pierre à 12h30'
    """
    return normalize_whitespace(s.translate(_PUNCTUATION))


def normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace into single spaces and trim both ends."""
    return " ".join(s.split())


def normalize(s: str) -> str:
    """Casefold, strip accents, replace punctuation with spaces, collapse whitespace.

    >>> normalize("  Jean-François   L'HÔPITAL! ")
    'jean francois l hopital'
    """
    return strip_punctuation(strip_accents(s.casefold()))
