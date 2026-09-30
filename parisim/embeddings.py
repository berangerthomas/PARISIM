"""Semantic similarity from text embeddings, with a pluggable encoder."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["DEFAULT_MODEL", "Embedding", "Encoder"]

#: Multilingual (50+ languages, French included), Apache-2.0, 0.22 GB.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Maps a list of texts to one vector per text (2-D array, list of lists...).
Encoder = Callable[[list[str]], Any]


class Embedding:
    """Cosine similarity of text embeddings, clipped to [0, 1].

    Texts can be encoded by:

    - a model name, loaded with fastembed (default backend, install the
      ``embeddings`` extra) or with sentence-transformers;
    - any function mapping a list of texts to one vector per text, to use an
      embedding API (OpenAI, Mistral...) or your own model.

    Vectors are cached, one per distinct text, and ``Matcher.compare_batch``
    encodes all the texts of a batch up front, *batch_size* texts per call.

    Usage::

        Matcher(methods=["embedding"])   # built-in, uses DEFAULT_MODEL
        m.add_method("minilm", Embedding("all-MiniLM-L6-v2",
                                         backend="sentence-transformers"))
        m.add_method("openai", Embedding(lambda texts: [
            d.embedding for d in client.embeddings.create(
                input=texts, model="text-embedding-3-small").data
        ]))

    Args:
        encoder: Model name, or ``list[str] -> vectors`` function.
        backend: ``"fastembed"`` or ``"sentence-transformers"``, the library
            loading *encoder* when it is a model name.
        batch_size: Maximum number of texts per encoder call.

    Examples:
        A toy encoder counting vowels, to show the mechanics:

        >>> def count_vowels(texts):
        ...     return [[t.count(v) for v in "aeiou"] for t in texts]
        >>> vowels = Embedding(count_vowels)
        >>> vowels("banana", "papaya")
        1.0
        >>> vowels("banana", "kiwi")
        0.0
    """

    category = "semantic"

    def __init__(
        self,
        encoder: str | Encoder = DEFAULT_MODEL,
        *,
        backend: str = "fastembed",
        batch_size: int = 256,
    ) -> None:
        if not isinstance(encoder, str) and not callable(encoder):
            kind = type(encoder).__name__
            raise TypeError(f"encoder must be a model name or a function, got {kind}")
        if backend not in _LOADERS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {list(_LOADERS)}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.encoder = encoder
        self.backend = backend
        self.batch_size = batch_size
        self._vectors: dict[str, npt.NDArray[np.float32]] = {}

    def __call__(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        self.prepare((a, b))
        cosine = float(self._vectors[a] @ self._vectors[b])
        return min(max(cosine, 0.0), 1.0)

    def prepare(self, texts: Iterable[str]) -> None:
        """Encode and cache the texts that are not cached yet."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if not missing:
            return
        np = _numpy()
        encode = self.encoder
        if isinstance(encode, str):
            encode = _load(self.backend, encode)
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            vectors = np.asarray(encode(chunk), dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(chunk):
                raise ValueError(
                    f"the encoder returned an array of shape {vectors.shape} for "
                    f"{len(chunk)} texts; expected one vector per text"
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
            self._vectors.update(zip(chunk, unit, strict=True))

    def __repr__(self) -> str:
        if callable(self.encoder):
            name = getattr(self.encoder, "__name__", type(self.encoder).__name__)
            return f"Embedding({name})"
        return f"Embedding({self.encoder!r}, backend={self.backend!r})"


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - numpy ships with every backend
        raise ImportError("Embeddings require numpy: pip install numpy") from exc
    return numpy


@functools.cache
def _load(backend: str, model: str) -> Encoder:
    """Load *model* with *backend*, once per process."""
    return _LOADERS[backend](model)


def _fastembed(model: str) -> Encoder:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            "The fastembed backend requires fastembed: pip install 'parisim[embeddings]'"
        ) from exc
    with warnings.catch_warnings():
        # fastembed warns that DEFAULT_MODEL now uses mean pooling: that is the
        # pooling it was trained with, so there is nothing to act on.
        warnings.filterwarnings("ignore", "The model .* now uses mean pooling")
        embedder = TextEmbedding(model_name=model)
    return lambda texts: list(embedder.embed(texts))


def _sentence_transformers(model: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The sentence-transformers backend requires sentence-transformers: "
            "pip install sentence-transformers"
        ) from exc
    embedder = SentenceTransformer(model)
    return lambda texts: embedder.encode(texts)


_LOADERS: dict[str, Callable[[str], Encoder]] = {
    "fastembed": _fastembed,
    "sentence-transformers": _sentence_transformers,
}
