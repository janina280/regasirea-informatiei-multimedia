"""Registru de modele: creare leneșă (încărcarea unui model durează și consumă memorie)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from .base import Encoder


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    languages: str
    factory: Callable[[], Encoder]


def _b32() -> Encoder:
    from .hf_clip import HFClipEncoder

    return HFClipEncoder("openai/clip-vit-base-patch32", key="clip-b32")


def _b16() -> Encoder:
    from .hf_clip import HFClipEncoder

    return HFClipEncoder("openai/clip-vit-base-patch16", key="clip-b16")


def _multi() -> Encoder:
    from .multilingual import MultilingualClipEncoder

    return MultilingualClipEncoder(key="clip-multilingual")


DEFAULT_SPECS = [
    ModelSpec("clip-b32", "CLIP ViT-B/32", "Rapid; baza de comparație (OpenAI).", "engleză", _b32),
    ModelSpec("clip-b16", "CLIP ViT-B/16", "Patch-uri mai mici: mai precis, mai lent.", "engleză", _b16),
    ModelSpec("clip-multilingual", "CLIP multilingv",
              "Imagini ViT-B/32, text multilingv distilat (50+ limbi).", "română și alte limbi", _multi),
]


class ModelRegistry:
    """Păstrează specificațiile modelelor și instanțele deja încărcate."""

    def __init__(self, specs: list[ModelSpec] | None = None, default: str = "clip-b32"):
        self._specs = {s.key: s for s in (specs or DEFAULT_SPECS)}
        self._loaded: dict[str, Encoder] = {}
        self._lock = threading.Lock()
        self.default = default if default in self._specs else next(iter(self._specs))

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def is_loaded(self, key: str) -> bool:
        return key in self._loaded

    def get(self, key: str | None = None) -> Encoder:
        key = key or self.default
        if key not in self._specs:
            raise KeyError(f"Model necunoscut: '{key}'. Disponibile: {', '.join(self._specs)}")
        with self._lock:
            if key not in self._loaded:
                enc = self._specs[key].factory()
                enc.key = key
                self._loaded[key] = enc
            return self._loaded[key]
