"""Embedding: backend-agnostic semantic similarity, tested with fake encoders."""

import re
import sys
import types
import warnings

import pytest

pytest.importorskip("numpy")

from parisim import Embedding, Matcher
from parisim.embeddings import DEFAULT_MODEL, _load

VECTORS = {
    "cat": [1.0, 0.0],
    "kitten": [0.8, 0.6],
    "dog": [0.0, 1.0],
    "anti-cat": [-1.0, 0.0],
    "void": [0.0, 0.0],
}


class CountingEncoder:
    """Look up VECTORS and record every call."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [VECTORS[t] for t in texts]


@pytest.fixture
def encoder():
    return CountingEncoder()


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    _load.cache_clear()
    yield
    _load.cache_clear()


# -- Similarity -------------------------------------------------------------------


def test_cosine_similarity(encoder):
    assert Embedding(encoder)("cat", "kitten") == pytest.approx(0.8)
    assert Embedding(encoder)("cat", "dog") == pytest.approx(0.0)


def test_negative_cosine_is_clipped_to_zero(encoder):
    assert Embedding(encoder)("cat", "anti-cat") == 0.0


def test_zero_vector_scores_zero(encoder):
    assert Embedding(encoder)("cat", "void") == 0.0


def test_identical_texts_score_one_without_encoding(encoder):
    assert Embedding(encoder)("cat", "cat") == 1.0
    assert encoder.calls == []


def test_vectors_are_cached(encoder):
    emb = Embedding(encoder)
    emb("cat", "kitten")
    emb("kitten", "cat")
    emb("cat", "dog")
    assert encoder.calls == [["cat", "kitten"], ["dog"]]


def test_prepare_encodes_in_chunks_of_batch_size(encoder):
    Embedding(encoder, batch_size=2).prepare(["cat", "kitten", "dog", "cat"])
    assert encoder.calls == [["cat", "kitten"], ["dog"]]


def test_encoder_returning_the_wrong_number_of_vectors_is_rejected():
    with pytest.raises(ValueError, match="one vector per text"):
        Embedding(lambda texts: [[1.0, 0.0]])("cat", "dog")


def test_invalid_arguments():
    with pytest.raises(ValueError, match="Unknown backend"):
        Embedding(backend="torch")
    with pytest.raises(TypeError, match="model name or a function"):
        Embedding(42)
    with pytest.raises(ValueError, match="batch_size"):
        Embedding(batch_size=0)


def test_repr():
    assert repr(Embedding()) == f"Embedding({DEFAULT_MODEL!r}, backend='fastembed')"
    assert repr(Embedding(CountingEncoder())) == "Embedding(CountingEncoder)"


# -- Backends ---------------------------------------------------------------------


def _fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_fastembed_backend_loads_the_model_once(monkeypatch):
    loaded = []

    class TextEmbedding:
        def __init__(self, model_name):
            loaded.append(model_name)

        def embed(self, texts):
            return (VECTORS[t] for t in texts)

    _fake_module(monkeypatch, "fastembed", TextEmbedding=TextEmbedding)
    assert Embedding("some/model")("cat", "kitten") == pytest.approx(0.8)
    assert Embedding("some/model")("cat", "dog") == pytest.approx(0.0)
    assert loaded == ["some/model"]


def test_fastembed_pooling_notice_is_silenced(monkeypatch):
    class TextEmbedding:
        def __init__(self, model_name):
            warnings.warn(f"The model {model_name} now uses mean pooling", stacklevel=2)

        def embed(self, texts):
            return (VECTORS[t] for t in texts)

    _fake_module(monkeypatch, "fastembed", TextEmbedding=TextEmbedding)
    assert Embedding(DEFAULT_MODEL)("cat", "kitten") == pytest.approx(0.8)


def test_sentence_transformers_backend(monkeypatch):
    class SentenceTransformer:
        def __init__(self, model):
            self.model = model

        def encode(self, texts):
            return [VECTORS[t] for t in texts]

    _fake_module(
        monkeypatch, "sentence_transformers", SentenceTransformer=SentenceTransformer
    )
    emb = Embedding("all-MiniLM-L6-v2", backend="sentence-transformers")
    assert emb("cat", "kitten") == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("backend", "module", "hint"),
    [
        ("fastembed", "fastembed", "parisim[embeddings]"),
        ("sentence-transformers", "sentence_transformers", "sentence-transformers"),
    ],
)
def test_missing_backend_raises_an_actionable_error(monkeypatch, backend, module, hint):
    monkeypatch.setitem(sys.modules, module, None)  # makes the import fail
    with pytest.raises(ImportError, match=re.escape(hint)):
        Embedding("some/model", backend=backend)("cat", "dog")


# -- With a Matcher ---------------------------------------------------------------


def test_embedding_is_opt_in():
    assert "embedding" not in Matcher().methods
    assert Matcher(methods="embedding").methods == ["embedding"]
    assert Matcher(categories="semantic").methods == ["embedding"]


def test_add_method_files_embeddings_under_semantic(encoder):
    m = Matcher(methods=[])
    m.add_method("toy", Embedding(encoder))
    assert m.compare("cat", "kitten").categories == {"toy": "semantic"}


def test_a_batch_is_encoded_up_front_in_one_call(encoder):
    m = Matcher(methods=[])
    m.add_method("toy", Embedding(encoder))
    batch = m.compare_one_to_many("cat", ["kitten", "dog", "cat"])
    assert encoder.calls == [["cat", "kitten", "dog"]]
    assert [r["toy"] for r in batch] == pytest.approx([0.8, 0.0, 1.0])


def test_encoder_failure_is_recorded_on_every_pair():
    def broken(texts):
        raise ConnectionError("API down")

    m = Matcher(methods="levenshtein")
    m.add_method("api", Embedding(broken))
    with pytest.warns(RuntimeWarning, match="'api' failed on 2 of 2"):
        batch = m.compare_batch([("a", "b"), ("c", "d")])
    assert all(r.errors == {"api": "ConnectionError: API down"} for r in batch)
    assert all("levenshtein" in r for r in batch)

    strict = Matcher(methods=[], strict=True)
    strict.add_method("api", Embedding(broken))
    with pytest.raises(ConnectionError):
        strict.compare("a", "b")
