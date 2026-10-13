"""Evaluare cantitativă: Recall@K, MRR, rang median, scalare cu mărimea galeriei, regăsire vs. generare.

Date de intrare: un CSV cu coloanele `image,caption` (mai multe rânduri pentru aceeași imagine = mai multe
descrieri de referință, ca în COCO sau Flickr). Galeria de texte conține toate descrierile din CSV.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

DEFAULT_KS = (1, 5, 10)


@dataclass
class PairsData:
    images: list[str]  # nume de fișier, în ordinea primei apariții
    captions: list[str]  # toate descrierile, grupate pe imagine
    owner: np.ndarray  # (n_captions,) indexul imaginii căreia îi aparține fiecare descriere


def load_pairs(csv_path: str | Path, max_images: int | None = None) -> PairsData:
    by_image: dict[str, list[str]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cap = " ".join(row["caption"].split())
            if cap:
                by_image.setdefault(row["image"], []).append(cap)
    names = list(by_image)[:max_images]
    captions, owner = [], []
    for i, n in enumerate(names):
        for c in by_image[n]:
            captions.append(c)
            owner.append(i)
    return PairsData(names, captions, np.asarray(owner, dtype=np.int64))


# ---------------------------------------------------------------- metrici
def _summ(ranks: np.ndarray, ks: Sequence[int]) -> dict:
    out = {f"R@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["MRR"] = float(np.mean(1.0 / ranks))
    out["median_rank"] = float(np.median(ranks))
    return out


def image_to_text_ranks(sim: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Pentru fiecare imagine: poziția (1 = cel mai bun) primei descrieri corecte în clasament."""
    n_img = sim.shape[0]
    ranks = np.empty(n_img, dtype=np.int64)
    for i in range(n_img):
        own = owner == i
        best_own = sim[i, own].max()
        ranks[i] = 1 + int(np.sum(sim[i, ~own] > best_own))
    return ranks


def text_to_image_ranks(sim: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Pentru fiecare descriere: poziția imaginii corecte printre toate imaginile."""
    own_sim = sim[owner, np.arange(sim.shape[1])]
    return 1 + np.sum(sim > own_sim[None, :], axis=0).astype(np.int64)


def compute_metrics(sim: np.ndarray, owner: np.ndarray, ks: Sequence[int] = DEFAULT_KS) -> dict:
    return {
        "image_to_text": _summ(image_to_text_ranks(sim, owner), ks),
        "text_to_image": _summ(text_to_image_ranks(sim, owner), ks),
    }


def recall_vs_gallery(sim: np.ndarray, owner: np.ndarray, sizes: Sequence[int], k: int = 1) -> list[dict]:
    """R@k pentru imagine→text când galeria conține doar primele n imagini și descrierile lor."""
    rows = []
    for n in sizes:
        if n > sim.shape[0]:
            continue
        mask = owner < n
        ranks = image_to_text_ranks(sim[:n][:, mask], owner[mask])
        rows.append({"n_images": int(n), "n_texts": int(mask.sum()), f"R@{k}": float(np.mean(ranks <= k))})
    return rows


# ---------------------------------------------------------------- rulare
def _load_images(names: Sequence[str], images_dir: Path) -> list[Image.Image]:
    out = []
    for n in names:
        with Image.open(images_dir / n) as im:
            out.append(im.convert("RGB"))
    return out


def evaluate_encoder(encoder, data: PairsData, images_dir: str | Path,
                     ks: Sequence[int] = DEFAULT_KS, sizes: Sequence[int] = ()) -> dict:
    images_dir = Path(images_dir)
    t0 = time.perf_counter()
    img_emb = encoder.encode_images(_load_images(data.images, images_dir))
    t1 = time.perf_counter()
    txt_emb = encoder.encode_texts(data.captions)
    t2 = time.perf_counter()
    sim = img_emb @ txt_emb.T
    res = compute_metrics(sim, data.owner, ks)
    res.update({
        "model": encoder.key,
        "model_name": encoder.model_name,
        "n_images": len(data.images),
        "n_texts": len(data.captions),
        "encode_ms_per_image": round((t1 - t0) * 1000 / len(data.images), 2),
        "encode_ms_per_text": round((t2 - t1) * 1000 / len(data.captions), 2),
    })
    if sizes:
        res["gallery_scaling"] = recall_vs_gallery(sim, data.owner, sizes)
    return res


def compare_retrieval_vs_generation(encoder, captioner, data: PairsData, images_dir: str | Path,
                                    n_images: int = 100) -> dict:
    """Scenariu „open-set”: pentru fiecare imagine, galeria NU conține propriile descrieri.

    Se compară titlul regăsit (cel mai apropiat text din descrierile celorlalte imagini) cu titlul generat
    de BLIP, prin două măsuri calculate cu CLIP:
      * image_text_sim: cosinus între imagine și titlu (CLIPScore brut);
      * ref_sim: cel mai mare cosinus text–text între titlu și descrierile de referință ale imaginii.
    """
    images_dir = Path(images_dir)
    n = min(n_images, len(data.images))
    imgs = _load_images(data.images[:n], images_dir)
    img_emb = encoder.encode_images(imgs)
    txt_emb = encoder.encode_texts(data.captions)

    retrieved, generated = [], []
    for i in range(n):
        own = data.owner == i
        sims = img_emb[i] @ txt_emb.T
        sims[own] = -np.inf
        retrieved.append(data.captions[int(np.argmax(sims))])
        generated.append(captioner.caption(imgs[i]))

    ret_emb, gen_emb = encoder.encode_texts(retrieved), encoder.encode_texts(generated)

    def score(cand_emb):
        its = np.einsum("ij,ij->i", img_emb[:n], cand_emb)
        refs = [float(np.max(txt_emb[data.owner == i] @ cand_emb[i])) for i in range(n)]
        return {"image_text_sim": float(np.mean(its)), "ref_sim": float(np.mean(refs))}

    return {
        "n_images": n,
        "retrieval_open_set": score(ret_emb),
        "generation_blip": score(gen_emb),
        "examples": [
            {"image": data.images[i], "retrieved": retrieved[i], "generated": generated[i],
             "reference": next(c for c, o in zip(data.captions, data.owner) if o == i)}
            for i in range(min(8, n))
        ],
    }


def run_comparison(registry, model_keys: Sequence[str], pairs_csv: str | Path, images_dir: str | Path,
                   out_path: str | Path, max_images: int | None = None, sizes: Sequence[int] = (),
                   captioner=None, gen_images: int = 100) -> dict:
    data = load_pairs(pairs_csv, max_images)
    report: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"pairs": str(pairs_csv), "n_images": len(data.images), "n_texts": len(data.captions)},
        "models": [],
    }
    for key in model_keys:
        enc = registry.get(key)
        report["models"].append(evaluate_encoder(enc, data, images_dir, sizes=sizes))
    if captioner is not None and model_keys:
        report["retrieval_vs_generation"] = {
            "encoder": model_keys[0],
            **compare_retrieval_vs_generation(registry.get(model_keys[0]), captioner, data, images_dir, gen_images),
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def markdown_table(report: dict) -> str:
    cols = ["R@1", "R@5", "R@10", "MRR", "median_rank"]
    lines = ["| Model | Direcție | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for m in report["models"]:
        for d, label in (("image_to_text", "imagine→text"), ("text_to_image", "text→imagine")):
            v = m[d]
            cells = [f"{v[c]:.3f}" if c != "median_rank" else f"{v[c]:.0f}" for c in cols]
            lines.append(f"| {m['model']} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
