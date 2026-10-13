"""Logica aplicației: leagă modelele, colecțiile și feedback-ul (independentă de FastAPI)."""

from __future__ import annotations

import hashlib
import io
import time

from PIL import Image

from .encoders import ModelRegistry
from .feedback import FeedbackStore
from .index import results_to_dicts
from .store import CollectionStore

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000


class InvalidImage(ValueError):
    pass


def open_image(data: bytes) -> Image.Image:
    """Validează și deschide o imagine încărcată de utilizator."""
    if not data:
        raise InvalidImage("Fișierul este gol.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage(f"Imaginea depășește {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise InvalidImage("Rezoluția imaginii este prea mare.")
        im.load()
        return im.convert("RGB")
    except InvalidImage:
        raise
    except Exception as e:  # PIL ridică tipuri variate de erori
        raise InvalidImage("Fișierul nu este o imagine validă (JPEG, PNG, WebP, GIF, BMP).") from e


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class RetrievalService:
    def __init__(self, store: CollectionStore, registry: ModelRegistry,
                 feedback: FeedbackStore | None = None, captioner=None):
        self.store = store
        self.registry = registry
        self.feedback = feedback
        self.captioner = captioner

    def search(self, image: Image.Image, collection: str, model: str | None = None,
               top_k: int = 5, min_score: float | None = None) -> dict:
        t0 = time.perf_counter()
        enc = self.registry.get(model)
        index = self.store.ensure_embeddings(collection, enc)
        t1 = time.perf_counter()
        vec = enc.encode_image(image)
        t2 = time.perf_counter()
        results = index.search(vec, top_k=top_k, min_score=min_score, scale=enc.logit_scale) if len(index) else []
        t3 = time.perf_counter()
        return {
            "model": enc.key,
            "collection": collection,
            "corpus_size": len(index),
            "results": results_to_dicts(results),
            "timing_ms": {
                "index_sync": round((t1 - t0) * 1000, 1),
                "image_encode": round((t2 - t1) * 1000, 1),
                "search": round((t3 - t2) * 1000, 2),
            },
        }

    def caption(self, image: Image.Image, collection: str, model: str | None = None,
                top_k: int = 5, min_score: float | None = None, generative: bool = False) -> dict:
        res = self.search(image, collection, model, top_k, min_score)
        hits = res["results"]
        if hits:
            best = hits[0]["text"]
            res["title"] = best[0].upper() + best[1:]
            res["alternatives"] = [h["text"] for h in hits[1:]]
        else:
            res["title"] = None
            res["alternatives"] = []
        if generative:
            if self.captioner is None:
                res["generative"] = {"error": "Modelul generativ nu este activat."}
            else:
                t = time.perf_counter()
                text = self.captioner.caption(image)
                res["generative"] = {"title": text[:1].upper() + text[1:],
                                     "time_ms": round((time.perf_counter() - t) * 1000, 1)}
        return res
