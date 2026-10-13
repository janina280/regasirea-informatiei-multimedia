"""Descarcă un subset din COCO val2017 (imagini + descrieri) pentru evaluare.

    python scripts/prepare_coco_subset.py --n 500

Rezultat în data/coco/: images/*.jpg și pairs.csv (coloane: image,caption; ~5 descrieri pe imagine).
Se descarcă o singură dată arhiva de adnotări COCO (~240 MB) și apoi doar cele N imagini alese.
Datele COCO: licență CC BY 4.0 pentru adnotări; imaginile provin din Flickr, fiecare cu licența proprie.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"
MEMBER = "annotations/captions_val2017.json"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="număr de imagini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/coco")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    ann_json = out / "captions_val2017.json"
    if not ann_json.exists():
        zip_path = out / "annotations_trainval2017.zip"
        print("Descarc adnotările COCO (~240 MB)…")
        download(ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            ann_json.write_bytes(z.read(MEMBER))
        zip_path.unlink()

    data = json.loads(ann_json.read_text(encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in data["images"]}
    caps: dict[int, list[str]] = {}
    for a in data["annotations"]:
        caps.setdefault(a["image_id"], []).append(" ".join(a["caption"].split()))

    ids = sorted(i for i in caps if i in files)
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]

    def fetch(i: int) -> tuple[int, bool]:
        try:
            download(IMG_URL.format(files[i]), out / "images" / files[i])
            return i, True
        except Exception as e:  # rețea
            print(f"  eșuat {files[i]}: {e}", file=sys.stderr)
            return i, False

    print(f"Descarc {len(ids)} imagini…")
    with ThreadPoolExecutor(8) as ex:
        ok = [i for i, good in ex.map(fetch, ids) if good]

    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption"])
        for i in ok:
            for c in caps[i]:
                w.writerow([files[i], c])
    print(f"Gata: {len(ok)} imagini, {sum(len(caps[i]) for i in ok)} descrieri în {out}/pairs.csv")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
