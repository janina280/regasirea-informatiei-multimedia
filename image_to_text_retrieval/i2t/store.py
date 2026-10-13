"""Colecții persistente de texte, cu vectori păstrați separat pentru fiecare model.

Structura pe disc (`<root>/collections/<nume>/`):
    texts.json            [{"id": 1, "text": "..."}, ...]   (ordinea = ordinea rândurilor din vectori)
    meta.json             {"next_id": N}
    emb_<model>.npy       (n_sincronizat, d) float32

Vectorii unui model se completează la nevoie (`ensure_embeddings`): doar textele noi sunt codificate.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .encoders.base import Encoder
from .index import TextIndex

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
MAX_TEXT_LEN = 500


class CollectionNotFound(KeyError):
    pass


class CollectionExists(ValueError):
    pass


@dataclass
class TextItem:
    id: int
    text: str


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


class CollectionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.base = self.root / "collections"
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---- colecții -------------------------------------------------------
    def _dir(self, name: str) -> Path:
        if not NAME_RE.match(name):
            raise ValueError("Nume invalid: folosește litere, cifre, '-' sau '_' (max. 64 de caractere).")
        return self.base / name

    def _require(self, name: str) -> Path:
        d = self._dir(name)
        if not (d / "texts.json").exists():
            raise CollectionNotFound(name)
        return d

    def list(self) -> list[dict]:
        out = []
        with self._lock:
            for d in sorted(p for p in self.base.iterdir() if (p / "texts.json").exists()):
                n = len(self._read_texts(d))
                models = sorted(f.stem[4:] for f in d.glob("emb_*.npy"))
                out.append({"name": d.name, "count": n, "indexed_models": models})
        return out

    def create(self, name: str) -> None:
        with self._lock:
            d = self._dir(name)
            if (d / "texts.json").exists():
                raise CollectionExists(name)
            d.mkdir(parents=True, exist_ok=True)
            self._write_texts(d, [])
            _atomic_write(d / "meta.json", json.dumps({"next_id": 1}))

    def delete(self, name: str) -> None:
        with self._lock:
            shutil.rmtree(self._require(name))

    # ---- texte ----------------------------------------------------------
    @staticmethod
    def _read_texts(d: Path) -> list[TextItem]:
        return [TextItem(**t) for t in json.loads((d / "texts.json").read_text(encoding="utf-8"))]

    @staticmethod
    def _write_texts(d: Path, items: list[TextItem]) -> None:
        _atomic_write(d / "texts.json",
                      json.dumps([{"id": i.id, "text": i.text} for i in items], ensure_ascii=False))

    def texts(self, name: str) -> list[TextItem]:
        with self._lock:
            return self._read_texts(self._require(name))

    def page(self, name: str, offset: int = 0, limit: int = 50, q: str = "") -> dict:
        items = self.texts(name)
        if q:
            ql = q.lower()
            items = [i for i in items if ql in i.text.lower()]
        return {"total": len(items), "items": [i.__dict__ for i in items[offset : offset + limit]]}

    def add_texts(self, name: str, texts: Sequence[str]) -> dict:
        """Adaugă texte noi; ignoră golurile, duplicatele și textele prea lungi."""
        with self._lock:
            d = self._require(name)
            items = self._read_texts(d)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            known = {i.text for i in items}
            added, skipped = 0, 0
            for t in texts:
                t = " ".join((t or "").split())
                if not t or t in known or len(t) > MAX_TEXT_LEN:
                    skipped += 1
                    continue
                items.append(TextItem(meta["next_id"], t))
                meta["next_id"] += 1
                known.add(t)
                added += 1
            self._write_texts(d, items)
            _atomic_write(d / "meta.json", json.dumps(meta))
            return {"added": added, "skipped": skipped, "total": len(items)}

    def delete_text(self, name: str, text_id: int) -> bool:
        with self._lock:
            d = self._require(name)
            items = self._read_texts(d)
            pos = next((i for i, it in enumerate(items) if it.id == text_id), None)
            if pos is None:
                return False
            del items[pos]
            self._write_texts(d, items)
            for f in d.glob("emb_*.npy"):
                emb = np.load(f)
                if emb.shape[0] > pos:
                    np.save(f, np.delete(emb, pos, axis=0))
            return True

    # ---- vectori --------------------------------------------------------
    def ensure_embeddings(self, name: str, encoder: Encoder) -> TextIndex:
        """Returnează indexul pentru (colecție, model), codificând doar textele încă neindexate."""
        with self._lock:
            d = self._require(name)
            items = self._read_texts(d)
            f = d / f"emb_{encoder.key}.npy"
            emb = np.load(f) if f.exists() else None
            if emb is not None and emb.shape[0] > len(items):  # defensiv: fișier desincronizat
                emb = None
            done = 0 if emb is None else emb.shape[0]
            if done < len(items):
                new = encoder.encode_texts([i.text for i in items[done:]])
                emb = new if emb is None else np.concatenate([emb, new], axis=0)
                np.save(f, emb)
            if emb is None:
                emb = np.zeros((0, 1), dtype=np.float32)
            return TextIndex([i.text for i in items], emb, encoder.model_name)
