"""Linia de comandă.

    python -m i2t.cli serve                                   # pornește API + interfața web
    python -m i2t.cli caption poza.jpg --collection demo-en   # titlu descriptiv din terminal
    python -m i2t.cli search poza.jpg --top-k 5
    python -m i2t.cli add --collection demo-en --texts data/captions.txt
    python -m i2t.cli eval --pairs data/coco/pairs.csv --images data/coco/images --models clip-b32,clip-b16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import PROJECT_ROOT, Settings, _seed
from .encoders import ModelRegistry
from .feedback import FeedbackStore
from .generative import LazyCaptioner
from .index import load_texts
from .service import InvalidImage, RetrievalService, open_image
from .store import CollectionExists, CollectionStore


def _service(args) -> RetrievalService:
    st = Settings()
    store = CollectionStore(args.storage or st.storage_dir)
    return RetrievalService(store, ModelRegistry(default=args.model), FeedbackStore(Path(args.storage or st.storage_dir) / "feedback.sqlite"), LazyCaptioner())


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("i2t.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_add(args) -> int:
    svc = _service(args)
    try:
        svc.store.create(args.collection)
    except CollectionExists:
        pass
    r = svc.store.add_texts(args.collection, load_texts(args.texts))
    print(f"Colecția '{args.collection}': +{r['added']} texte noi, {r['skipped']} ignorate, total {r['total']}.")
    return 0


def _run(args, fn):
    svc = _service(args)
    _seed(svc.store, Settings().data_dir)
    try:
        with open(args.image, "rb") as f:
            im = open_image(f.read())
    except FileNotFoundError:
        print(f"Eroare: imaginea '{args.image}' nu există.", file=sys.stderr)
        return None, 1
    except InvalidImage as e:
        print(f"Eroare: {e}", file=sys.stderr)
        return None, 1
    return fn(svc, im), 0


def cmd_search(args) -> int:
    out, code = _run(args, lambda s, im: s.search(im, args.collection, args.model, args.top_k, args.min_score))
    if code:
        return code
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if not out["results"]:
        print("Niciun text nu depășește pragul de similaritate.")
    for r in out["results"]:
        print(f"{r['rank']:>2}. [{r['score']:.3f} | {r['probability']:6.1%}] {r['text']}")
    return 0


def cmd_caption(args) -> int:
    out, code = _run(args, lambda s, im: s.caption(im, args.collection, args.model, args.top_k,
                                                   args.min_score, args.generative))
    if code:
        return code
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if out["title"] is None:
        print("Nu am găsit un titlu suficient de apropiat. Adaugă mai multe texte candidate.")
    else:
        print(f"Titlu: {out['title']}  (scor {out['results'][0]['score']:.3f})")
        for a in out["alternatives"]:
            print(f"  - {a}")
    if args.generative:
        print(f"Generat (BLIP): {out['generative'].get('title') or out['generative'].get('error')}")
    return 0


def cmd_eval(args) -> int:
    from .evaluation import markdown_table, run_comparison

    registry = ModelRegistry()
    sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else []
    report = run_comparison(
        registry, args.models.split(","), args.pairs, args.images, args.out,
        max_images=args.max_images, sizes=sizes,
        captioner=LazyCaptioner() if args.generative else None, gen_images=args.gen_images,
    )
    print(markdown_table(report))
    print(f"\nRaport salvat în {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="i2t", description="Image-to-Text Retrieval")
    p.add_argument("--storage", default=None, help="folderul de stocare (implicit: ./storage)")
    p.add_argument("--model", default="clip-b32", help="clip-b32 | clip-b16 | clip-multilingual")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="pornește serverul web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("add", help="adaugă texte într-o colecție (o creează dacă lipsește)")
    a.add_argument("--collection", required=True)
    a.add_argument("--texts", required=True)
    a.set_defaults(func=cmd_add)

    for name, fn in (("search", cmd_search), ("caption", cmd_caption)):
        q = sub.add_parser(name)
        q.add_argument("image")
        q.add_argument("--collection", default="demo-en")
        q.add_argument("--top-k", type=int, default=5)
        q.add_argument("--min-score", type=float, default=None)
        q.add_argument("--json", action="store_true")
        if name == "caption":
            q.add_argument("--generative", action="store_true", help="compară și cu BLIP")
        q.set_defaults(func=fn)

    e = sub.add_parser("eval", help="evaluează modelele pe un set imagine–descrieri")
    e.add_argument("--pairs", required=True, help="CSV cu coloanele image,caption")
    e.add_argument("--images", required=True, help="folderul cu imagini")
    e.add_argument("--models", default="clip-b32,clip-b16")
    e.add_argument("--out", default=str(PROJECT_ROOT / "results" / "eval.json"))
    e.add_argument("--max-images", type=int, default=None)
    e.add_argument("--sizes", default="100,250,500,1000", help="mărimi de galerie pentru curba de scalare")
    e.add_argument("--generative", action="store_true", help="adaugă comparația regăsire vs. BLIP")
    e.add_argument("--gen-images", type=int, default=100)
    e.set_defaults(func=cmd_eval)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
