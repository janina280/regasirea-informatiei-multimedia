import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from i2t.index import TextIndex, load_texts  # noqa: E402


class FakeEncoder:
    """Vectori one-hot: fiecare text își are propria axă, deci rezultatele sunt previzibile."""

    model_name = "fake"

    def __init__(self, vocab):
        self.vocab = list(vocab)

    def encode_texts(self, texts):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, self.vocab.index(t)] = 1.0
        return out


TEXTS = ["a cat", "a dog", "a car", "a tree"]


@pytest.fixture
def index():
    return TextIndex.build(TEXTS, FakeEncoder(TEXTS))


def test_search_orders_by_cosine(index):
    q = np.array([0.1, 0.9, 0.0, 0.2], dtype=np.float32)  # cel mai aproape de "a dog"
    res = index.search(q, top_k=3)
    assert [r.text for r in res] == ["a dog", "a tree", "a cat"]
    assert [r.rank for r in res] == [1, 2, 3]
    assert res[0].score > res[1].score > res[2].score
    assert abs(sum(r.probability for r in index.search(q, top_k=4)) - 1.0) < 1e-5


def test_top_k_is_clamped(index):
    assert len(index.search(np.ones(4), top_k=100)) == 4
    assert len(index.search(np.ones(4), top_k=0)) == 1


def test_min_score_filters(index):
    q = np.array([1.0, 0.0, 0.0, 0.0])
    res = index.search(q, top_k=4, min_score=0.5)
    assert [r.text for r in res] == ["a cat"]


def test_query_scale_does_not_matter(index):
    q = np.array([0.2, 0.7, 0.1, 0.0])
    a = index.search(q, top_k=4)
    b = index.search(q * 50, top_k=4)
    assert [r.text for r in a] == [r.text for r in b]
    assert a[0].score == pytest.approx(b[0].score, abs=1e-5)


def test_dimension_mismatch_raises(index):
    with pytest.raises(ValueError, match="Dimensiune"):
        index.search(np.ones(7))


def test_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = TextIndex.load(tmp_path / "idx")
    assert loaded.texts == TEXTS
    assert loaded.model_name == "fake"
    np.testing.assert_allclose(loaded.embeddings, index.embeddings)


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="build"):
        TextIndex.load(tmp_path / "nope")


def test_build_rejects_empty():
    with pytest.raises(ValueError):
        TextIndex.build([], FakeEncoder(TEXTS))


def test_load_texts_txt_dedupes_and_strips(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("a cat \n\n a dog\na cat\n", encoding="utf-8")
    assert load_texts(f) == ["a cat", "a dog"]


def test_load_texts_csv_and_jsonl(tmp_path):
    c = tmp_path / "t.csv"
    c.write_text("id,caption\n1,a cat\n2,a dog\n", encoding="utf-8")
    assert load_texts(c) == ["a cat", "a dog"]
    j = tmp_path / "t.jsonl"
    j.write_text(json.dumps({"text": "a car"}) + "\n" + json.dumps("a tree") + "\n", encoding="utf-8")
    assert load_texts(j) == ["a car", "a tree"]
