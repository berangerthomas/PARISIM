"""Matcher: run several similarity methods over pairs of strings."""

from __future__ import annotations

import difflib
import math
import numbers
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from parisim.methods import BUILTINS, OPT_IN, Similarity
from parisim.result import BatchResult, MatchResult

#: A preprocessing step: ``str -> str``.
Preprocessor = Callable[[str], str]

# Floating-point slack accepted around [0, 1] before a score is deemed invalid.
_TOLERANCE = 1e-9


class Matcher:
    """Compare strings with several similarity methods at once.

    Args:
        methods: Name(s) of the methods to run. ``None`` (default) runs every
            built-in method except the opt-in ``"embedding"``.
        categories: Category name(s) of the methods to run, used when
            *methods* is ``None``.
        preprocess: ``str -> str`` function(s) applied in order to both strings
            before comparing them, e.g. ``normalize`` or ``str.lower``.
        strict: Raise as soon as a method fails. By default the error is
            recorded in ``MatchResult.errors``, a warning is emitted, and the
            other methods still run.

    Examples:
        >>> m = Matcher()
        >>> m.compare("hello", "helo")["levenshtein"]
        0.8
        >>> Matcher(categories="token").compare("red car", "car red")["token_sort"]
        1.0
        >>> m = Matcher("levenshtein", preprocess=str.lower)
        >>> m.compare("ABC", "abc")["levenshtein"]
        1.0
    """

    def __init__(
        self,
        methods: str | Sequence[str] | None = None,
        categories: str | Sequence[str] | None = None,
        preprocess: Preprocessor | Sequence[Preprocessor] | None = None,
        strict: bool = False,
    ) -> None:
        if preprocess is None:
            preprocess = ()
        elif callable(preprocess):
            preprocess = (preprocess,)
        for fn in preprocess:
            if not callable(fn):
                raise TypeError(f"preprocess expects functions, got {fn!r}")
        self._preprocess = tuple(preprocess)
        self._strict = strict
        self._methods = _select(methods, categories)

    # -- Public API -------------------------------------------------------------

    @property
    def methods(self) -> list[str]:
        """Names of the methods this matcher runs."""
        return list(self._methods)

    def compare(self, a: str, b: str) -> MatchResult:
        """Compare two strings with every method.

        Returns:
            MatchResult with one score per method that returned a value.

        Raises:
            TypeError: If *a* or *b* is not a string (e.g. a NaN from pandas).
        """
        return self._run([_check_pair((a, b))])[0]

    def compare_batch(self, pairs: Iterable[tuple[str, str]]) -> BatchResult:
        """Compare many pairs, e.g. ``zip(df["name_a"], df["name_b"])``.

        Raises:
            TypeError: If an item is not a pair of strings; the message gives
                its position.
        """
        return BatchResult(self._run(_check_pairs(pairs)))

    def compare_one_to_many(self, query: str, candidates: Iterable[str]) -> BatchResult:
        """Compare *query* with each of *candidates*."""
        return BatchResult(self._run(_check_pairs((query, c) for c in candidates)))

    def add_method(self, name: str, fn: Similarity, category: str | None = None) -> None:
        """Add a method to this matcher, or replace the one called *name*.

        Args:
            name: Method name, the key of its score in ``MatchResult.scores``.
            fn: ``(a, b) -> float | None`` returning a similarity in [0, 1], or
                ``None`` when it does not apply to the pair. Any callable
                works: a lambda, ``functools.partial(tfidf_cosine, ...)``, an
                ``Embedding(...)``...
            category: Category label; defaults to ``fn.category`` when defined
                (``"semantic"`` for an ``Embedding``), else ``"custom"``.
        """
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        if category is None:
            category = str(getattr(fn, "category", "custom"))
        self._methods[name] = (category, fn)

    @staticmethod
    def available_methods() -> list[str]:
        """Names of the built-in methods."""
        return list(BUILTINS)

    @staticmethod
    def available_categories() -> list[str]:
        """Categories of the built-in methods."""
        return list(dict.fromkeys(category for category, _ in BUILTINS.values()))

    # -- Internals --------------------------------------------------------------

    def _run(self, pairs: list[tuple[str, str]]) -> list[MatchResult]:
        cleaned = [(self._clean(a), self._clean(b)) for a, b in pairs]
        categories = {name: category for name, (category, _) in self._methods.items()}
        results = [MatchResult(a, b, categories=categories) for a, b in pairs]
        for name, (_, fn) in self._methods.items():
            self._score(name, fn, cleaned, results)
            failures = [r.errors[name] for r in results if name in r.errors]
            if failures:
                # stacklevel=3 blames the user's line: every public compare* method
                # calls _run directly.
                warnings.warn(
                    f"method {name!r} failed on {len(failures)} of {len(results)} "
                    f"pair(s): {failures[0]}. See MatchResult.errors, or pass "
                    "strict=True to raise instead.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return results

    def _score(
        self,
        name: str,
        fn: Similarity,
        cleaned: list[tuple[str, str]],
        results: list[MatchResult],
    ) -> None:
        """Run one method over every pair, recording scores and errors."""
        prepare = getattr(fn, "prepare", None)
        if prepare is not None:  # e.g. Embedding: encode every text in a few calls
            try:
                prepare([text for pair in cleaned for text in pair])
            except Exception as exc:
                if self._strict:
                    raise
                for result in results:
                    result.errors[name] = _describe(exc)
                return
        for result, (a, b) in zip(results, cleaned, strict=True):
            try:
                score = fn(a, b)
                if score is not None:
                    result.scores[name] = _as_score(score)
            except Exception as exc:
                if self._strict:
                    raise
                result.errors[name] = _describe(exc)

    def _clean(self, s: str) -> str:
        for fn in self._preprocess:
            s = fn(s)
            if not isinstance(s, str):
                raise TypeError(
                    f"preprocess function {_name(fn)} returned "
                    f"{type(s).__name__}, expected str"
                )
        return s

    def __len__(self) -> int:
        return len(self._methods)

    def __repr__(self) -> str:
        args = f"methods={self.methods}"
        if self._preprocess:
            args += f", preprocess=[{', '.join(_name(fn) for fn in self._preprocess)}]"
        return f"Matcher({args})"


def _select(
    methods: str | Sequence[str] | None,
    categories: str | Sequence[str] | None,
) -> dict[str, tuple[str, Similarity]]:
    """Pick the built-in methods matching the Matcher arguments."""
    if methods is not None:
        names = _as_list(methods)
        _check_known(names, Matcher.available_methods(), "method")
        return {name: BUILTINS[name] for name in names}
    if categories is not None:
        wanted = _as_list(categories)
        _check_known(wanted, Matcher.available_categories(), "category")
        return {n: m for n, m in BUILTINS.items() if m[0] in wanted}
    return {n: m for n, m in BUILTINS.items() if n not in OPT_IN}


def _as_list(names: str | Sequence[str]) -> list[str]:
    return [names] if isinstance(names, str) else list(names)


def _check_known(names: list[str], known: list[str], kind: str) -> None:
    for name in names:
        if name not in known:
            close = difflib.get_close_matches(name, known, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"Unknown {kind} {name!r}.{hint} Available: {known}")


def _check_pairs(pairs: Iterable[object]) -> list[tuple[str, str]]:
    """Check every pair with ``_check_pair``; an error gives the pair's position."""
    checked = []
    for i, pair in enumerate(pairs):
        try:
            checked.append(_check_pair(pair))
        except TypeError as exc:
            raise TypeError(f"pair {i}: {exc}") from None
    return checked


def _check_pair(pair: object) -> tuple[str, str]:
    """Return *pair* as ``(a, b)``, or raise TypeError if it is not two strings."""
    if isinstance(pair, str):
        raise TypeError(f"expected a pair of strings, got the string {pair!r}")
    try:
        a, b = cast("Iterable[object]", pair)
    except (TypeError, ValueError):
        raise TypeError(f"expected a pair of strings, got {pair!r}") from None
    if isinstance(a, str) and isinstance(b, str):
        return a, b
    bad = b if isinstance(a, str) else a
    hint = " (missing value? drop or fill NaN/None first)" if _is_missing(bad) else ""
    raise TypeError(f"expected str, got {type(bad).__name__} {bad!r}{hint}")


def _is_missing(value: object) -> bool:
    """True for None, NaN and pandas.NA."""
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or type(value).__name__ == "NAType"
    )


def _as_score(value: object) -> float:
    """Return *value* as a float, if it is a similarity in [0, 1]."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"expected a number, got {type(value).__name__} {value!r}")
    score = float(value)
    if not -_TOLERANCE <= score <= 1 + _TOLERANCE:  # also rejects NaN
        raise ValueError(f"expected a similarity in [0, 1], got {value!r}")
    return min(max(score, 0.0), 1.0)


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _name(fn: Callable[..., object]) -> str:
    return getattr(fn, "__name__", repr(fn))
