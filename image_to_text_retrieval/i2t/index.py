"""Indexul textelor precalculate și căutarea celor mai apropiate texte pentru un vector de imagine."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

EMB_FILE = "embeddings.npy"
META_FILE = "meta.json"
TEXT_COLUMNS = ("text", "caption", "title", "description", "comment")


class TextEncoder(Protocol):
    model_name: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class SearchResult:
    rank: int
    text: str
    score: float  # similaritate cosinus, în [-1, 1]
    probability: float  # softmax(scale * scor) peste tot indexul


def load_texts(path: str | Path) -> list[str]:
    """Citește texte din .txt (câte unul pe linie), .csv (coloană text/caption/title/...) sau .jsonl.

    Liniile goale se ignoră, iar duplicatele se elimină păstrând ordinea.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    raw: list[str] = []
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return []
        col = next((c for c in TEXT_COLUMNS if c in rows[0]), next(iter(rows[0])))
        raw = [r[col] for r in rows]
    elif suffix in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw.append(obj if isinstance(obj, str) else next(obj[c] for c in TEXT_COLUMNS if c in obj))
    else:
        raw = path.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


class TextIndex:
    """Vectorii textelor (L2-normalizați) și textele corespunzătoare."""

    def __init__(self, texts: Sequence[str], embeddings: np.ndarray, model_name: str = ""):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(texts) != embeddings.shape[0]:
            raise ValueError("embeddings trebuie să aibă forma (n_texte, dim) și n_texte = len(texts)")
        self.texts = list(texts)
        self.embeddings = embeddings
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.texts)

    @classmethod
    def build(cls, texts: Sequence[str], encoder: TextEncoder) -> "TextIndex":
        if not texts:
            raise ValueError("Nu există texte de indexat.")
        return cls(texts, encoder.encode_texts(list(texts)), getattr(encoder, "model_name", ""))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMB_FILE, self.embeddings)
        (directory / META_FILE).write_text(
            json.dumps({"model_name": self.model_name, "texts": self.texts}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "TextIndex":
        directory = Path(directory)
        if not (directory / EMB_FILE).exists() or not (directory / META_FILE).exists():
            raise FileNotFoundError(
                f"Nu există un index în '{directory}'. Rulează mai întâi comanda 'build'."
            )
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(meta["texts"], np.load(directory / EMB_FILE), meta.get("model_name", ""))

    def search(
        self,
        image_vec: np.ndarray,
        top_k: int = 5,
        min_score: float | None = None,
        scale: float = 100.0,
    ) -> list[SearchResult]:
        """Cele mai apropiate `top_k` texte pentru vectorul unei imagini (similaritate cosinus)."""
        q = np.asarray(image_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Dimensiune incompatibilă: imagine {q.shape[0]} vs index {self.embeddings.shape[1]} "
                "(indexul a fost construit cu alt model?)"
            )
        q = q / max(float(np.linalg.norm(q)), 1e-12)

        scores = self.embeddings @ q
        logits = scale * scores
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        k = max(1, min(top_k, len(self)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        results = [
            SearchResult(rank=0, text=self.texts[i], score=float(scores[i]), probability=float(probs[i]))
            for i in top
            if min_score is None or scores[i] >= min_score
        ]
        for r, res in enumerate(results, 1):
            res.rank = r
        return results


def results_to_dicts(results: Sequence[SearchResult]) -> list[dict]:
    return [asdict(r) for r in results]
