import numpy as np
import pytest

from i2t.store import CollectionExists, CollectionNotFound, CollectionStore


@pytest.fixture
def store(tmp_path):
    s = CollectionStore(tmp_path)
    s.create("c1")
    return s


def test_create_list_delete(store):
    assert [c["name"] for c in store.list()] == ["c1"]
    with pytest.raises(CollectionExists):
        store.create("c1")
    store.delete("c1")
    assert store.list() == []
    with pytest.raises(CollectionNotFound):
        store.texts("c1")


@pytest.mark.parametrize("bad", ["", "../x", "a b", "-x", "a" * 80])
def test_invalid_names_rejected(store, bad):
    with pytest.raises(ValueError):
        store.create(bad)


def test_add_texts_cleans_and_dedupes(store):
    r = store.add_texts("c1", ["  a  red   car ", "a red car", "", "x" * 600, "a blue sea"])
    assert r == {"added": 2, "skipped": 3, "total": 2}
    assert [t.text for t in store.texts("c1")] == ["a red car", "a blue sea"]


def test_ids_are_never_reused(store):
    store.add_texts("c1", ["a", "b"])
    store.delete_text("c1", 2)
    store.add_texts("c1", ["c"])
    assert [t.id for t in store.texts("c1")] == [1, 3]


def test_incremental_embeddings_encode_only_new_texts(store, fake_encoder):
    store.add_texts("c1", ["a red car", "a blue sea"])
    idx = store.ensure_embeddings("c1", fake_encoder)
    assert len(idx) == 2 and fake_encoder.calls == 2
    store.ensure_embeddings("c1", fake_encoder)
    assert fake_encoder.calls == 2  # nimic nou de codificat
    store.add_texts("c1", ["a green field"])
    idx = store.ensure_embeddings("c1", fake_encoder)
    assert len(idx) == 3 and fake_encoder.calls == 3


def test_delete_text_keeps_embeddings_aligned(store, fake_encoder):
    store.add_texts("c1", ["a red car", "a blue sea", "a green field"])
    store.ensure_embeddings("c1", fake_encoder)
    assert store.delete_text("c1", 2)
    idx = store.ensure_embeddings("c1", fake_encoder)
    assert idx.texts == ["a red car", "a green field"]
    np.testing.assert_allclose(idx.embeddings, fake_encoder.encode_texts(idx.texts))
    assert not store.delete_text("c1", 99)


def test_per_model_embeddings_are_separate(store, fake_encoder):
    from tests.conftest import FakeEncoder

    store.add_texts("c1", ["a red car"])
    store.ensure_embeddings("c1", fake_encoder)
    store.ensure_embeddings("c1", FakeEncoder("other"))
    assert store.list()[0]["indexed_models"] == ["fake", "other"]


def test_page_and_filter(store):
    store.add_texts("c1", [f"text {i}" for i in range(30)] + ["needle here"])
    assert store.page("c1", 0, 10)["total"] == 31
    assert len(store.page("c1", 0, 10)["items"]) == 10
    assert store.page("c1", q="needle")["total"] == 1


def test_persistence_across_instances(tmp_path):
    a = CollectionStore(tmp_path)
    a.create("c1")
    a.add_texts("c1", ["hello"])
    b = CollectionStore(tmp_path)
    assert [t.text for t in b.texts("c1")] == ["hello"]
