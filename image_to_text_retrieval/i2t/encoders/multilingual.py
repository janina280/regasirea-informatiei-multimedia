"""CLIP cu text multilingv (inclusiv română).

Imaginile sunt codificate de CLIP ViT-B/32. Textul este codificat de un model multilingv
(`clip-ViT-B-32-multilingual-v1`), distilat astfel încât vectorii textelor în peste 50 de limbi
să cadă în același spațiu cu vectorii imaginilor CLIP ViT-B/32.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .base import Encoder, normalize, pick_device

IMAGE_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


class MultilingualClipEncoder(Encoder):
    def __init__(self, key: str = "clip-multilingual", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.key = key
        self.model_name = f"{IMAGE_MODEL} + {TEXT_MODEL}"
        self.device = device or pick_device()
        self.image_model = SentenceTransformer(IMAGE_MODEL, device=self.device)
        self.text_model = SentenceTransformer(TEXT_MODEL, device=self.device)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        emb = self.text_model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return normalize(emb)

    def encode_images(self, images: Iterable[Image.Image], batch_size: int = 16) -> np.ndarray:
        imgs = [im.convert("RGB") for im in images]
        emb = self.image_model.encode(
            imgs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return normalize(emb)
