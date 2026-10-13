"""API REST (FastAPI) + servirea interfeței web.

Pornire:  uvicorn i2t.api:app --reload      sau      python -m i2t.cli serve
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .encoders import ModelRegistry
from .feedback import FeedbackStore
from .generative import LazyCaptioner
from .index import load_texts
from .service import InvalidImage, RetrievalService, image_hash, open_image
from .store import CollectionExists, CollectionNotFound, CollectionStore

WEB_DIR = Path(__file__).parent / "web"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_TEXT_BYTES = 5 * 1024 * 1024
SEED = {"demo-en": "captions.txt", "demo-ro": "captions_ro.txt"}


@dataclass
class Settings:
    storage_dir: Path = Path(os.environ.get("I2T_STORAGE", PROJECT_ROOT / "storage"))
    data_dir: Path = Path(os.environ.get("I2T_DATA", PROJECT_ROOT / "data"))
    results_dir: Path = Path(os.environ.get("I2T_RESULTS", PROJECT_ROOT / "results"))
    seed_demo: bool = os.environ.get("I2T_SEED", "1") == "1"


class CollectionIn(BaseModel):
    name: str


class TextsIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=5000)


class FeedbackIn(BaseModel):
    collection: str
    model: str
    image_hash: str
    text: str
    rank: int = Field(ge=1)
    score: float
    relevant: bool


def _seed(store: CollectionStore, data_dir: Path) -> None:
    if store.list():
        return
    for name, fname in SEED.items():
        f = data_dir / fname
        if f.exists():
            store.create(name)
            store.add_texts(name, load_texts(f))


def create_app(service: RetrievalService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if service is None:
        store = CollectionStore(settings.storage_dir)
        service = RetrievalService(
            store, ModelRegistry(), FeedbackStore(settings.storage_dir / "feedback.sqlite"), LazyCaptioner()
        )
    if settings.seed_demo:
        _seed(service.store, settings.data_dir)

    app = FastAPI(title="Image-to-Text Retrieval", version="1.0")
    svc = service

    def _check(collection: str) -> None:
        try:
            svc.store.texts(collection)
        except CollectionNotFound:
            raise HTTPException(404, f"Colecția '{collection}' nu există.")
        except ValueError as e:
            raise HTTPException(422, str(e))

    def _model(key: Optional[str]) -> Optional[str]:
        if key and key not in {s.key for s in svc.registry.specs()}:
            raise HTTPException(422, f"Model necunoscut: '{key}'.")
        return key

    # ---------------------------------------------------------- meta
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/models")
    def models():
        return [
            {"key": s.key, "label": s.label, "description": s.description, "languages": s.languages,
             "loaded": svc.registry.is_loaded(s.key), "default": s.key == svc.registry.default}
            for s in svc.registry.specs()
        ]

    # ---------------------------------------------------------- colecții
    @app.get("/api/collections")
    def collections():
        return svc.store.list()

    @app.post("/api/collections", status_code=201)
    def create_collection(body: CollectionIn):
        try:
            svc.store.create(body.name)
        except CollectionExists:
            raise HTTPException(409, f"Colecția '{body.name}' există deja.")
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"name": body.name, "count": 0}

    @app.delete("/api/collections/{name}", status_code=204)
    def delete_collection(name: str):
        try:
            svc.store.delete(name)
        except CollectionNotFound:
            raise HTTPException(404, f"Colecția '{name}' nu există.")
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/collections/{name}/texts")
    def list_texts(name: str, offset: int = 0, limit: int = 50, q: str = ""):
        _check(name)
        return svc.store.page(name, max(offset, 0), min(max(limit, 1), 200), q)

    @app.post("/api/collections/{name}/texts")
    def add_texts(name: str, body: TextsIn):
        _check(name)
        return svc.store.add_texts(name, body.texts)

    @app.post("/api/collections/{name}/upload")
    async def upload_texts(name: str, file: UploadFile = File(...)):
        _check(name)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".txt", ".csv", ".jsonl", ".ndjson"):
            raise HTTPException(422, "Format acceptat: .txt, .csv sau .jsonl.")
        data = await file.read()
        if len(data) > MAX_UPLOAD_TEXT_BYTES:
            raise HTTPException(413, "Fișierul depășește 5 MB.")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
        try:
            texts = load_texts(tmp.name)
        except Exception:
            raise HTTPException(422, "Fișierul nu a putut fi interpretat (verifică formatul și coloanele).")
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        return svc.store.add_texts(name, texts)

    @app.delete("/api/collections/{name}/texts/{text_id}", status_code=204)
    def delete_text(name: str, text_id: int):
        _check(name)
        if not svc.store.delete_text(name, text_id):
            raise HTTPException(404, "Textul nu există.")

    # ---------------------------------------------------------- regăsire
    async def _read_image(image: UploadFile):
        data = await image.read()
        try:
            return open_image(data), image_hash(data)
        except InvalidImage as e:
            raise HTTPException(422, str(e))

    @app.post("/api/search")
    async def search(image: UploadFile = File(...), collection: str = Form(...), model: Optional[str] = Form(None),
                     top_k: int = Form(5), min_score: Optional[float] = Form(None)):
        _check(collection)
        model = _model(model)
        im, h = await _read_image(image)
        out = svc.search(im, collection, model, max(1, min(top_k, 50)), min_score)
        out["image_hash"] = h
        return out

    @app.post("/api/caption")
    async def caption(image: UploadFile = File(...), collection: str = Form(...), model: Optional[str] = Form(None),
                      top_k: int = Form(5), min_score: Optional[float] = Form(None),
                      generative: bool = Form(False)):
        _check(collection)
        model = _model(model)
        im, h = await _read_image(image)
        out = svc.caption(im, collection, model, max(1, min(top_k, 50)), min_score, generative)
        out["image_hash"] = h
        return out

    # ---------------------------------------------------------- feedback & evaluare
    @app.post("/api/feedback", status_code=201)
    def feedback(body: FeedbackIn):
        if svc.feedback is None:
            raise HTTPException(503, "Feedback-ul nu este activat.")
        svc.feedback.add(**body.model_dump())
        return {"ok": True}

    @app.get("/api/feedback/stats")
    def feedback_stats():
        if svc.feedback is None:
            raise HTTPException(503, "Feedback-ul nu este activat.")
        return svc.feedback.stats()

    @app.get("/api/evaluation")
    def evaluation():
        f = settings.results_dir / "eval.json"
        if not f.exists():
            raise HTTPException(404, "Nu există rezultate. Rulează: python -m i2t.cli eval ...")
        return json.loads(f.read_text(encoding="utf-8"))

    # ---------------------------------------------------------- interfața web
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    """`uvicorn i2t.api:app` -> aplicația se creează la prima cerere, nu la import."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
