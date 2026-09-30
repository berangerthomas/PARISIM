# PARISIM

**Pa**ttern **R**ecognition & **I**ntelligent **Sim**ilarity: compare strings with many similarity methods at once, from typos to meaning, and keep the ones that suit your data.

```python
from parisim import Matcher

Matcher().compare("voiture rouge", "automobile rouge")
```

```
MatchResult('voiture rouge' vs 'automobile rouge')
  jaro_winkler   0.754
  partial_ratio  0.750
  indel_ratio    0.621
  token_set      0.621
  levenshtein    0.500
  soundex        0.500
  metaphone      0.500
  token_sort     0.414
  tfidf_cosine   0.336
```

- Methods for typos, word order, truncation, pronunciation and meaning, all scoring in [0, 1].
- Batches go straight to pandas: one row per pair, one column per method.
- Light core (`rapidfuzz`, `jellyfish`); pandas, matplotlib and embeddings are optional.
- Semantic similarity with any embedding model or API.
- Python 3.10 to 3.14, fully typed.

## Installation

```bash
pip install parisim
pip install "parisim[all]"
```

Extras: `pandas` for `to_pandas()`, `viz` for `plot()`, `embeddings` for the fastembed backend, `all` for everything.

## Choosing a method

| Strings that differ by... | Method | Example |
|---|---|---|
| typos | `levenshtein`, `indel_ratio` | Dupont / Dupond |
| typos in short strings, names | `jaro_winkler` | Martha / Marhta |
| characters of equal-length codes | `hamming` | 75001 / 75002 |
| truncation, abbreviation | `partial_ratio` | Société Générale / Société Générale SA |
| word order | `token_sort` | Jean Dupont / Dupont Jean |
| extra or repeated words | `token_set` | Jean Dupont / M. Jean Dupont |
| shared words | `tfidf_cosine` | rue de la Paix / 12 rue de la Paix |
| pronunciation, English rules | `soundex`, `metaphone` | Philip / Filip |
| meaning | `embedding` (opt-in) | voiture / automobile |

`levenshtein` counts a substitution as one edit, `indel_ratio` (the classic `fuzz.ratio`) as two. `hamming` returns no score for strings of different lengths. `soundex` needs the same first letter (Robert ~ Rupert), `metaphone` does not (Philip ~ Filip). Phonetic methods compare word by word; words without Latin letters, such as numbers, must match exactly.

Methods fall into four categories: `character` (levenshtein, indel_ratio, hamming, jaro_winkler, partial_ratio), `token` (token_sort, token_set, tfidf_cosine), `phonetic` (soundex, metaphone) and `semantic` (embedding).

**Each method has its own scale.** Two unrelated names, "Jean Dupont" and "Marie Curie", get 0.49 from `jaro_winkler` but 0 from `tfidf_cosine`. Compare a method's scores across pairs and choose a threshold per method, never compare the scores of two methods.

## Usage

### Select methods

```python
Matcher(methods=["levenshtein", "jaro_winkler"])
Matcher(categories="token")
Matcher.available_methods()
```

`Matcher()` runs every built-in method except `embedding`.

### Preprocess

```python
from parisim import Matcher, normalize, strip_accents

Matcher(preprocess=normalize)  # casefold, accents, punctuation, spaces
Matcher(preprocess=[str.lower, strip_accents])  # any str -> str functions, in order
```

`normalize("  Jean-François   L'HÔPITAL! ")` gives `'jean francois l hopital'`. `strip_punctuation` and `normalize_whitespace` are also available. Results keep the original strings.

### Batches and pandas

```python
batch = m.compare_batch(zip(df["name_a"], df["name_b"]))
scores = batch.to_pandas()  # str1, str2, then one column per method

m.compare_one_to_many("paris", ["paris", "pari", "lyon"])
```

A method that does not apply to a pair gets NaN. With pairs labelled as duplicates or not, see which methods tell them apart best:

```python
scores["duplicate"] = df["duplicate"]
scores.groupby("duplicate").mean(numeric_only=True).T
```

### Charts

```python
m.compare("Société Générale", "Societe Generale SA").plot()  # one bar per method
batch.plot()  # distribution of each method's scores
```

Both return a matplotlib `Axes` and accept `ax=` to draw into your own figure.

### Semantic similarity

Character and word methods cannot tell that *voiture* and *automobile* mean the same thing; embeddings can. The `embedding` method is opt-in: it requires the `embeddings` extra and downloads its model on first use.

```python
Matcher(["levenshtein", "embedding"]).compare("voiture rouge", "automobile rouge")
# embedding 0.996, levenshtein 0.500
```

The default model is [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (50+ languages, Apache-2.0, 0.22 GB). `Embedding` accepts any other model, or any function turning a list of texts into vectors:

```python
from parisim import Embedding

m.add_method("bge", Embedding("BAAI/bge-small-en-v1.5"))  # fastembed
m.add_method("minilm", Embedding("all-MiniLM-L6-v2", backend="sentence-transformers"))
m.add_method("api", Embedding(encode))  # e.g. calls OpenAI or Mistral
```

Vectors are cached, and a batch is encoded in a few large calls. Check the licence of the model you pick: some, such as jina-embeddings-v3, forbid commercial use.

### Custom methods

```python
m.add_method("same_initial", lambda a, b: float(a[:1] == b[:1]))
```

A method returns a similarity in [0, 1], or `None` when it does not apply to a pair. Built-in methods are plain functions in `parisim.methods`, to use directly or adapt with `functools.partial`:

```python
from functools import partial
from parisim.methods import tfidf_cosine

m.add_method("tfidf_1_letter", partial(tfidf_cosine, token_pattern=r"(?u)\b\w+\b"))
```

### Errors

- Inputs must be strings. `NaN` or `None` raise a `TypeError` giving the pair's position: drop or fill missing values first.
- A method that raises, or returns something other than a number in [0, 1], does not stop the others: the error goes to `result.errors` and a warning is emitted. `Matcher(strict=True)` raises instead.

See [examples/demo.ipynb](https://github.com/berangerthomas/PARISIM/blob/main/examples/demo.ipynb) for a guided tour.

## Development

```bash
uv sync
uv run pytest --cov   # also runs the doctests and the demo notebook
uv run ruff check . && uv run ruff format .
uv run mypy parisim
```

## License

MIT, see [LICENSE](https://github.com/berangerthomas/PARISIM/blob/main/LICENSE).
