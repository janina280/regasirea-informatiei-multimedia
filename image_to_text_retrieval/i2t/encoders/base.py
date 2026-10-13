"""Interfața comună a codificatoarelor imagine/text care partajează un spațiu vectorial."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalizare L2 pe rânduri; cu vectori normalizați, produsul scalar = similaritate cosinus."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype(np.float32)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder(ABC):
    """Un model care știe să transforme texte și imagini în vectori comparabili între ei."""

    #: cheie scurtă stabilă (folosită în fișierele de index și în API)
    key: str = ""
    #: identificatorul modelului (ex. numele de pe Hugging Face)
    model_name: str = ""
    #: temperatura pentru transformarea scorurilor cosinus în probabilități
    logit_scale: float = 100.0

    @abstractmethod
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """(n, d) float32, L2-normalizat."""

    @abstractmethod
    def encode_images(self, images: Iterable[Image.Image], batch_size: int = 16) -> np.ndarray:
        """(n, d) float32, L2-normalizat."""

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.encode_images([image])[0]
