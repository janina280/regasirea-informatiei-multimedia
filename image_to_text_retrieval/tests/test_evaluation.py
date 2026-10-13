import csv

import numpy as np
import pytest
from PIL import Image

from i2t.evaluation import (
    compute_metrics, image_to_text_ranks, load_pairs, recall_vs_gallery, run_comparison, text_to_image_ranks,
)

# 3 imagini; descrierile 0,1 -> img0; 2 -> img1; 3,4 -> img2
OWNER = np.array([0, 0, 1, 2, 2])


def test_perfect_retrieval():
    sim = np.zeros((3, 5))
    for j, o in enumerate(OWNER):
        sim[o, j] = 1.0
    m = compute_metrics(sim, OWNER)
    assert m["image_to_text"]["R@1"] == 1.0 and m["image_to_text"]["MRR"] == 1.0
    assert m["text_to_image"]["R@1"] == 1.0 and m["text_to_image"]["median_rank"] == 1.0


def test_ranks_known_values():
    sim = np.array([
        [0.1, 0.9, 0.5, 0.2, 0.0],  # img0: cea mai bună proprie = 0.9 (rang 1)
        [0.8, 0.7, 0.3, 0.6, 0.1],  # img1: propria 0.3, mai mari: 0.8,0.7,0.6 -> rang 4
        [0.0, 0.0, 0.0, 0.4, 0.4],  # img2: propria 0.4, nimic mai mare -> rang 1
    ])
    assert list(image_to_text_ranks(sim, OWNER)) == [1, 4, 1]
    # text 2 (a imaginii 1): scor propriu 0.3; imaginile cu scor mai mare pe coloana 2: img0 (0.5) -> rang 2
    assert text_to_image_ranks(sim, OWNER)[2] == 2


def test_ties_do_not_inflate_ranks():
    sim = np.ones((3, 5))
    assert list(image_to_text_ranks(sim, OWNER)) == [1, 1, 1]


def test_recall_vs_gallery_is_monotone_for_hard_distractor():
    sim = np.array([
        [0.9, 0.9, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.1, 0.1],
        [0.1, 0.1, 0.95, 0.9, 0.9],  # img2 e confundată cu textul 2 al imaginii 1? nu: img2 are propriile 0.9
    ])
    rows = recall_vs_gallery(sim, OWNER, [1, 2, 3, 99])
    assert [r["n_images"] for r in rows] == [1, 2, 3]
    assert rows[0]["R@1"] == 1.0


def _make_dataset(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for name, color, words in [("r.png", (255, 0, 0), ["a red car", "red flower"]),
                               ("g.png", (0, 255, 0), ["green grass"]),
                               ("b.png", (0, 0, 255), ["blue sea", "a blue sky"])]:
        Image.new("RGB", (16, 16), color).save(img_dir / name)
        rows += [(name, w) for w in words]
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption"])
        w.writerows(rows)
    return pairs, img_dir


def test_load_pairs_groups_by_image(tmp_path):
    pairs, _ = _make_dataset(tmp_path)
    d = load_pairs(pairs)
    assert d.images == ["r.png", "g.png", "b.png"]
    assert list(d.owner) == [0, 0, 1, 2, 2]
    assert len(load_pairs(pairs, max_images=2).images) == 2


def test_run_comparison_end_to_end(tmp_path, registry):
    pairs, img_dir = _make_dataset(tmp_path)
    from tests.conftest import FakeCaptioner

    out = tmp_path / "res" / "eval.json"
    rep = run_comparison(registry, ["fake", "fake2"], pairs, img_dir, out, sizes=[2, 3], captioner=FakeCaptioner(), gen_images=3)
    assert out.exists()
    assert rep["models"][0]["image_to_text"]["R@1"] == 1.0
    assert rep["models"][0]["gallery_scaling"][-1]["n_images"] == 3
    assert rep["retrieval_vs_generation"]["n_images"] == 3
    assert "retrieval_open_set" in rep["retrieval_vs_generation"]
