import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from i2t.api import Settings, create_app
from i2t.feedback import FeedbackStore
from i2t.service import RetrievalService
from i2t.store import CollectionStore

from .conftest import FakeCaptioner


def png(color, size=(24, 24)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def client(tmp_path, registry):
    store = CollectionStore(tmp_path / "s")
    svc = RetrievalService(store, registry, FeedbackStore(tmp_path / "s" / "fb.sqlite"), FakeCaptioner())
    app = create_app(svc, Settings(storage_dir=tmp_path / "s", data_dir=tmp_path, results_dir=tmp_path / "r", seed_demo=False))
    return TestClient(app)


@pytest.fixture
def col(client):
    assert client.post("/api/collections", json={"name": "demo"}).status_code == 201
    r = client.post("/api/collections/demo/texts", json={"texts": ["a red car", "a green field", "a blue sea"]})
    assert r.json()["added"] == 3
    return "demo"


def search(client, color, **form):
    data = {"collection": "demo", "model": "fake", "top_k": "3", **form}
    return client.post("/api/search", data=data, files={"image": ("x.png", png(color), "image/png")})


def test_basics(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert [m["key"] for m in client.get("/api/models").json()] == ["fake", "fake2"]
    assert "Image-to-Text" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_collection_crud_and_errors(client, col):
    assert client.post("/api/collections", json={"name": "demo"}).status_code == 409
    assert client.post("/api/collections", json={"name": "bad name"}).status_code == 422
    assert client.get("/api/collections/nope/texts").status_code == 404
    page = client.get("/api/collections/demo/texts").json()
    assert page["total"] == 3
    assert client.delete(f"/api/collections/demo/texts/{page['items'][0]['id']}").status_code == 204
    assert client.delete("/api/collections/demo/texts/999").status_code == 404
    assert client.delete("/api/collections/demo").status_code == 204
    assert client.get("/api/collections").json() == []


def test_upload_texts(client, col):
    r = client.post("/api/collections/demo/upload", files={"file": ("t.txt", b"a red bus\na red car\n", "text/plain")})
    assert r.json()["added"] == 1
    assert client.post("/api/collections/demo/upload", files={"file": ("t.exe", b"x", "text/plain")}).status_code == 422
    bad = client.post("/api/collections/demo/upload", files={"file": ("t.jsonl", b"{not json", "text/plain")})
    assert bad.status_code == 422


@pytest.mark.parametrize("color,expected", [((250, 5, 5), "a red car"), ((5, 250, 5), "a green field"), ((5, 5, 250), "a blue sea")])
def test_search_returns_semantically_closest_text(client, col, color, expected):
    body = search(client, color).json()
    assert body["results"][0]["text"] == expected
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert body["corpus_size"] == 3 and "timing_ms" in body


def test_search_validation(client, col):
    assert search(client, (1, 2, 3), collection="nope").status_code == 404
    assert search(client, (1, 2, 3), model="nope").status_code == 422
    bad = client.post("/api/search", data={"collection": "demo"}, files={"image": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 422
    empty = client.post("/api/search", data={"collection": "demo"}, files={"image": ("x.png", b"", "image/png")})
    assert empty.status_code == 422


def test_min_score_and_empty_collection(client, col):
    assert search(client, (250, 5, 5), min_score="0.999").json()["results"][0]["text"] == "a red car"
    client.post("/api/collections", json={"name": "empty"})
    r = search(client, (250, 5, 5), collection="empty")
    assert r.status_code == 200 and r.json()["results"] == []


def test_caption_title_alternatives_and_generative(client, col):
    r = client.post("/api/caption", data={"collection": "demo", "model": "fake", "top_k": "3", "generative": "true"},
                    files={"image": ("x.png", png((250, 5, 5)), "image/png")}).json()
    assert r["title"] == "A red car"
    assert len(r["alternatives"]) == 2
    assert r["generative"]["title"] == "A generated caption"


def test_caption_without_hits_has_no_title(client, col):
    r = client.post("/api/caption", data={"collection": "demo", "min_score": "1.1"},
                    files={"image": ("x.png", png((250, 5, 5)), "image/png")}).json()
    assert r["title"] is None and r["alternatives"] == []


def test_feedback_roundtrip(client, col):
    body = search(client, (250, 5, 5)).json()
    top = body["results"][0]
    for rel in (True, True, False):
        assert client.post("/api/feedback", json={"collection": "demo", "model": "fake", "image_hash": body["image_hash"],
                                                  "text": top["text"], "rank": 1, "score": top["score"], "relevant": rel}).status_code == 201
    st = client.get("/api/feedback/stats").json()
    assert st["total"] == 3 and st["relevant"] == 2
    assert client.post("/api/feedback", json={"collection": "demo"}).status_code == 422


def test_evaluation_endpoint(client, tmp_path):
    assert client.get("/api/evaluation").status_code == 404
    (tmp_path / "r").mkdir(exist_ok=True)
    (tmp_path / "r" / "eval.json").write_text(json.dumps({"models": []}))
    assert client.get("/api/evaluation").json() == {"models": []}
