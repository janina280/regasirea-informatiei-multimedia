"""CLIP din Hugging Face Transformers (text engleză)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .base import Encoder, normalize, pick_device


def _features(out):
    """transformers 4.x întoarce un tensor; 5.x un obiect cu vectorii proiectați în `pooler_output`."""
    return out if hasattr(out, "cpu") else out.pooler_output


class HFClipEncoder(Encoder):
    def __init__(self, model_name: str, key: str = "", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self._torch = torch
        self.model_name = model_name
        self.key = key or model_name
        self.device = device or pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.logit_scale = float(self.model.logit_scale.exp().item())

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inputs = self.processor(
                    text=list(texts[i : i + batch_size]),
                    return_tensors="pt", padding=True, truncation=True, max_length=77,
                ).to(self.device)
                out.append(_features(self.model.get_text_features(**inputs)).cpu().numpy())
        return normalize(np.concatenate(out, axis=0))

    def encode_images(self, images: Iterable[Image.Image], batch_size: int = 16) -> np.ndarray:
        images = [im.convert("RGB") for im in images]
        out = []
        with self._torch.no_grad():
            for i in range(0, len(images), batch_size):
                inputs = self.processor(images=images[i : i + batch_size], return_tensors="pt").to(self.device)
                out.append(_features(self.model.get_image_features(**inputs)).cpu().numpy())
        return normalize(np.concatenate(out, axis=0))
