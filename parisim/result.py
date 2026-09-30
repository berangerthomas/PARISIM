"""Result containers: one pair (MatchResult) or many pairs (BatchResult)."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


@dataclass(repr=False)
class MatchResult:
    """Similarity scores of one pair of strings.

    Attributes:
        str1: First string, as given (before preprocessing).
        str2: Second string, as given.
        scores: ``{method: score}`` for every method that returned a score.
        categories: ``{method: category}`` for every method that ran.
        errors: ``{method: message}`` for every method that failed.

    Examples:
        >>> from parisim import Matcher
        >>> result = Matcher(["levenshtein", "token_sort"]).compare("red car", "car red")
        >>> result
        MatchResult('red car' vs 'car red')
          token_sort   1.000
          levenshtein  0.143
        >>> result["token_sort"]
        1.0
    """

    str1: str
    str2: str
    scores: dict[str, float] = field(default_factory=dict)
    categories: Mapping[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain, JSON-serializable dictionary."""
        return {
            "str1": self.str1,
            "str2": self.str2,
            "scores": dict(self.scores),
            "categories": dict(self.categories),
            "errors": dict(self.errors),
        }

    def plot(self, ax: Axes | None = None) -> Axes:
        """Bar chart of the scores, grouped by category. Requires matplotlib.

        Args:
            ax: Axes to draw on; a new figure is created if omitted.

        Returns:
            The Axes, to customize or save the chart.
        """
        from parisim._plotting import plot_scores

        return plot_scores(self, ax=ax)

    def __getitem__(self, method: str) -> float:
        return self.scores[method]

    def __contains__(self, method: object) -> bool:
        return method in self.scores

    def __repr__(self) -> str:
        head = f"MatchResult({_short(self.str1)!r} vs {_short(self.str2)!r}"
        if not self.scores and not self.errors:
            return f"{head}, no scores)"
        width = max(map(len, [*self.scores, *self.errors]))
        lines = [f"{head})"]
        for method, score in sorted(self.scores.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {method:<{width}}  {score:.3f}")
        for method, message in self.errors.items():
            lines.append(f"  {method:<{width}}  failed: {message}")
        return "\n".join(lines)


@dataclass(repr=False)
class BatchResult:
    """Scores of many pairs, one MatchResult per pair.

    Iterate over it, index it, or turn it into a DataFrame with ``to_pandas()``.
    """

    results: list[MatchResult] = field(default_factory=list)

    def to_pandas(self) -> pd.DataFrame:
        """One row per pair: ``str1``, ``str2``, then one column per method.

        A method that returned no score for a pair (not applicable, or failed)
        gets NaN. Requires pandas.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_pandas() requires pandas: pip install pandas") from exc
        data: dict[str, list[Any]] = {
            "str1": [r.str1 for r in self.results],
            "str2": [r.str2 for r in self.results],
        }
        for method in self._methods():
            data[method] = [r.scores.get(method, math.nan) for r in self.results]
        return pd.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain, JSON-serializable dictionary."""
        return {"pairs": [r.to_dict() for r in self.results]}

    def plot(self, ax: Axes | None = None) -> Axes:
        """Box plot of each method's scores over the batch. Requires matplotlib.

        Shows how each method spreads its scores on your data, which differs a
        lot from one method to another: a threshold rarely transfers.

        Args:
            ax: Axes to draw on; a new figure is created if omitted.

        Returns:
            The Axes, to customize or save the chart.
        """
        from parisim._plotting import plot_distributions

        return plot_distributions(self, ax=ax)

    def _methods(self) -> list[str]:
        """Methods that ran on at least one pair, in matcher order."""
        return list(dict.fromkeys(m for r in self.results for m in r.categories))

    def __iter__(self) -> Iterator[MatchResult]:
        return iter(self.results)

    def __getitem__(self, index: int) -> MatchResult:
        return self.results[index]

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        text = f"BatchResult({len(self)} pairs, {len(self._methods())} methods"
        failed = sum(1 for r in self.results if r.errors)
        if failed:
            text += f", {failed} pairs with errors"
        return text + ")"


def _short(s: str, width: int = 40) -> str:
    """Truncate *s* to *width* characters for display."""
    return s if len(s) <= width else s[: width - 1] + "…"
