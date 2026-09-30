# Changelog

## [0.3.0] - 2026-09-30

**The project is renamed PARISIM**: `pip install parisim`, `import parisim`. The `prism` name is taken on PyPI.

### Removed
- Methods redundant with a kept one (Spearman rho >= 0.98 on 1,500 French name and address pairs): `exact_match`, `damerau_levenshtein`, `jaro`, `sequence_matcher`, `double_metaphone` and `match_rating_codex`. `sequence_matcher` was also asymmetric, and scored two near-identical 350-character texts 0.108.
- The `Metaphone` dependency, last released in 2016.
- `MatchResult.best`: the method with the highest raw score means nothing, since each method has its own scale (`jaro` "won" 81% of random name pairs). Also `ranked()`, `by_category()`, and `BatchResult.summary()`, `averages()`, `by_category()` and `categories`: use `to_pandas()`.
- `register` and `SimilarityMethod`: `Matcher.add_method` is now the one way to add a method.
- `lowercase`, `casefold`, `identity` and `compose`: pass `str.lower` or `str.casefold` directly.
- The public `prism.visualization` module: use `result.plot()` and `batch.plot()`.

### Changed
- Categories are now `character`, `token`, `phonetic` and `semantic`.
- Built-in methods are plain functions in `parisim.methods`.
- The semantic method, renamed `embedding` (was `jina_v3`), is opt-in: `Matcher()` no longer runs it. Its default model is `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 0.22 GB) instead of `jina-embeddings-v3` (CC-BY-NC-4.0, which forbids commercial use; 2.29 GB). Negative cosines are clipped to 0.
- Unknown method or category names raise `ValueError` with a suggestion, instead of `KeyError`.
- `plot()` returns a matplotlib `Axes`, accepts `ax=` and no longer calls `plt.show()`. Bars share one color; the batch chart shows each method's score distribution instead of its mean.
- `strip_punctuation` replaces punctuation and symbols with spaces: "l'avion" gives "l avion", not "lavion".
- `MatchResult`'s repr lists every score.

### Added
- `Embedding`: semantic similarity from any model name (fastembed or sentence-transformers) or any `list[str] -> vectors` function, such as an embedding API. Vectors are cached and batches are encoded up front.
- `BatchResult.to_pandas()`.
- `normalize`: casefold, strip accents, punctuation to spaces, collapse whitespace.
- `Matcher` accepts a single method name, category or preprocessing function.
- Input validation: a non-string such as NaN or None raises a `TypeError` giving the pair's position.
- Score validation: a method must return a number in [0, 1]; scores are converted to `float`.
- A warning when a method fails, pointing to `MatchResult.errors`.
- `pandas` extra; the `viz` extra now requires matplotlib 3.10.
- CI: mypy --strict, coverage threshold, the demo notebook (moved to `examples/`) runs in the test suite, Dependabot.

### Fixed
- `token_set` and `tfidf_cosine` scored some identical strings 0: empty strings, punctuation, one-letter words.
- Phonetic methods scored "route 66" vs "route 67" 1.0: words without Latin letters, such as numbers, must now match exactly.
- `result.plot(show=False)` returned `None`, losing the figure.
- Missing values from pandas got 0.0 from some methods and errors from others.
- `strip_accents` turned "ガス" (gas) into "カス" (dregs) and decomposed Korean syllables; `strip_punctuation` deleted Devanagari vowel signs.
- A NaN custom score could become `best`, and a `numpy.float32` one broke the JSON export.
- The test suite could not be imported on Python 3.10 (`tomllib`).

## [0.2.0] - 2026-08-24

### Removed
- **`scikit-learn` is no longer a core dependency.** `tfidf_cosine` now ships a dependency-free implementation validated bit-for-bit against `TfidfVectorizer` + `cosine_similarity` (209 cases in the test suite, max deviation 2.2e-16). The core install drops from ~100 MB to ~3 MB.
- Dropped the unused `asttokens` dependency.

### Changed
- **`requires-python` lowered from `>=3.14` to `>=3.10`.** The codebase never used any 3.11+ syntax; the old floor excluded almost every Python user.
- **`levenshtein` now computes Levenshtein.** It previously called `rapidfuzz.fuzz.ratio`, which is the *indel* ratio — substitutions cost two edits there, not one, so `("kitten", "sitting")` returned 0.615 instead of 0.571. The old behaviour is still available as the new `indel_ratio` method.
- **Phonetic methods compare token by token.** Soundex and friends encode a whole string into one short code, so `soundex("voiture rouge") == "V366"` — every token after the first was invisible. Scores are now the ratio of phonetically matching tokens, which also makes them graded instead of binary.
- `jaro_winkler` moved from `jellyfish` to `rapidfuzz` (identical results, ~10x faster). `hamming` and `damerau_levenshtein` likewise.
- `Matcher(methods=[])` and `Matcher(categories=[])` now select *no* methods. Empty sequences were falsy and silently fell through to "all methods".
- `Matcher.add_method(..., category=...)` is now honoured by `by_category()` and by the plots; the label used to be discarded in favour of `"custom"`.
- `Matcher(categories=["typo"])` raises `KeyError` instead of silently producing an empty matcher.
- `MatchResult.best` raises a message naming the pair and the failures instead of a bare `max() iterable argument is empty`.
- `BatchResult.averages()` divides by the number of pairs where a method actually produced a score, so partially-applicable methods such as `hamming` are no longer diluted toward zero.
- `plot_comparison` / `plot_averages` return the Figure and accept `show=False`, so they can be embedded or saved instead of only displayed.
- Missing optional dependencies now raise an actionable `ImportError` naming the extra to install.

### Added
- `indel_ratio` (`edit_distance`) — the classic `fuzz.ratio` / fuzzywuzzy ratio.
- `jaro` (`sequence`) — Jaro similarity without the Winkler prefix bonus.
- `MatchResult.errors` — methods that raise are recorded instead of vanishing silently, and `Matcher(strict=True)` re-raises them.
- `Matcher.compare_one_to_many(query, candidates)`, `Matcher.methods`, `len(matcher)`.
- `BatchResult.by_category()`, `BatchResult.to_dict()`, `BatchResult.categories`.
- `casefold` and `identity` preprocessors; `TfidfCosine(token_pattern=..., lowercase=...)` and `JinaEmbedding(model_name=...)` are configurable.
- `py.typed` marker — type hints are now visible to downstream type checkers.
- Test suite (82 tests) and a CI matrix over Python 3.10–3.14.
- Full PyPI metadata: license expression, author, keywords, classifiers, project URLs.

### Fixed
- `import prism` no longer imports scikit-learn, scipy, numpy or matplotlib: **1477 ms to 13 ms**. `tfidf_cosine` is ~77x faster.
- `double_metaphone("", "")` returned 0.0 while every other phonetic method returned 1.0.
- `match_rating_codex` raised on tokens containing digits; the exception was swallowed and the method silently disappeared from the results.

## [0.1.0] - 2026-02-10

### Added
- Initial release of PRISM (Pattern Recognition & Intelligent Similarity Matcher).
- Core matching algorithms: Levenshtein, Jaro-Winkler, TF-IDF, Phonetic methods.
- Support for token-based matching (Set, Sort).
- Visualization module with `matplotlib`.
- FastEmbed integration for semantic similarity.
