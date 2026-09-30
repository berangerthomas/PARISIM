"""Charts for MatchResult and BatchResult. Requires matplotlib."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from parisim.result import _short

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from parisim.result import BatchResult, MatchResult

# One series, one hue: bars share the reference palette's first slot. Categories
# are told apart by position and headers, not color.
_SERIES = "#2a78d6"
_WASH = "#2a78d626"  # the series hue at 15% opacity, for box fills
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_MUTED = "#898781"

# A row is (category, method); method is None for the category's header row.
Row = tuple[str, str | None]


def plot_scores(result: MatchResult, ax: Axes | None = None) -> Axes:
    """One horizontal bar per method, grouped by category."""
    rows = _rows({m: [s] for m, s in result.scores.items()}, result.categories)
    title = f"{_short(result.str1)!r} vs {_short(result.str2)!r}"
    ax = _prepare_axes(ax, rows, title)
    ys = [y for y, (_, method) in enumerate(rows) if method]
    scores = [result.scores[method] for _, method in rows if method]
    ax.barh(ys, scores, height=0.6, color=_SERIES)
    for y, score in zip(ys, scores, strict=True):
        ax.text(score + 0.01, y, f"{score:.2f}", va="center", fontsize=8, color=_MUTED)
    return ax


def plot_distributions(batch: BatchResult, ax: Axes | None = None) -> Axes:
    """One box per method: the spread of its scores over the batch."""
    categories: dict[str, str] = {}
    for result in batch:
        categories.update(result.categories)
    values = {m: [r.scores[m] for r in batch if m in r.scores] for m in categories}
    rows = _rows(values, categories)
    ax = _prepare_axes(ax, rows, f"Score distribution over {len(batch)} pairs")
    ys = [y for y, (_, method) in enumerate(rows) if method]
    if ys:
        ax.boxplot(
            [values[method] for _, method in rows if method],
            positions=ys,
            orientation="horizontal",
            widths=0.6,
            manage_ticks=False,
            patch_artist=True,
            boxprops={"facecolor": _WASH, "edgecolor": _SERIES},
            medianprops={"color": _SERIES, "linewidth": 2},
            whiskerprops={"color": _SERIES},
            capprops={"color": _SERIES},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": _SERIES,
                "markeredgecolor": "none",
                "alpha": 0.5,
            },
        )
    return ax


def _rows(
    values: Mapping[str, Sequence[float]], categories: Mapping[str, str]
) -> list[Row]:
    """Rows top to bottom: each category's header, then its methods by
    decreasing median score. Categories keep the matcher's order."""
    groups: dict[str, list[str]] = {}
    for method, category in categories.items():
        if values.get(method):
            groups.setdefault(category, []).append(method)
    rows: list[Row] = []
    for category, methods in groups.items():
        rows.append((category, None))
        methods.sort(key=lambda m: -statistics.median(values[m]))
        rows.extend((category, m) for m in methods)
    return rows


def _prepare_axes(ax: Axes | None, rows: list[Row], title: str) -> Axes:
    """Create or reuse Axes, and lay out a 0-1 score axis with one line per row."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib: pip install 'parisim[viz]'"
        ) from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 1.2 + 0.3 * len(rows)), layout="constrained")
    ax.set_title(title, loc="left")
    ax.set_xlim(0, 1.08)  # room for the value labels
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Similarity")
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(axis="y", length=0)
    if not rows:
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No scores to plot",
            ha="center",
            color=_MUTED,
            transform=ax.transAxes,
        )
        return ax
    ax.set_yticks(range(len(rows)), [method or "" for _, method in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)  # first row on top
    for y, (category, method) in enumerate(rows):
        if method is None:
            ax.text(
                0, y, category, va="center", fontsize=9, fontweight="bold", color=_MUTED
            )
    return ax
