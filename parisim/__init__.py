"""PARISIM — Pattern Recognition & Intelligent Similarity."""

from parisim.embeddings import Embedding
from parisim.matcher import Matcher
from parisim.preprocessing import (
    normalize,
    normalize_whitespace,
    strip_accents,
    strip_punctuation,
)
from parisim.result import BatchResult, MatchResult

__version__ = "0.3.0"

__all__ = [
    "BatchResult",
    "Embedding",
    "MatchResult",
    "Matcher",
    "__version__",
    "normalize",
    "normalize_whitespace",
    "strip_accents",
    "strip_punctuation",
]
