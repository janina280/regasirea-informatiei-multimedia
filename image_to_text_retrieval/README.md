# Image-to-Text Retrieval

Sistem de regăsire **imagine → text** bazat pe CLIP, cu API REST, interfață web, evaluare cantitativă
și comparație cu un model generativ (BLIP). Aplicație principală: **generarea de titluri descriptive
pentru imagini**, prin regăsirea celui mai apropiat text dintr-o colecție.

```
imagine ──► codificator vizual ──► vector ┐
                                          ├─ similaritate cosinus ─► top-K texte ─► titlu + alternative
colecție de texte ─► codificator text ─► vectori (precalculați, persistenți) ┘
```

## Ce conține

| Componentă | Fișier | Rol |
|---|---|---|
| Codificatoare | `i2t/encoders/` | CLIP ViT-B/32, ViT-B/16 și CLIP multilingv (română); registru cu încărcare leneșă |
| Index | `i2t/index.py` | căutare exactă prin produs scalar pe vectori normalizați, scoruri cosinus + probabilități |
| Colecții persistente | `i2t/store.py` | texte + vectori pe model; indexare incrementală (se codifică doar textele noi) |
| Serviciu | `i2t/service.py` | logica de căutare/titrare, validarea imaginilor, cronometrare pe etape |
| API REST | `i2t/api.py` | FastAPI; documentație interactivă la `/docs` |
| Interfață web | `i2t/web/` | căutare cu comparare de modele, gestionarea colecțiilor, evaluare, feedback |
| Evaluare | `i2t/evaluation.py` | Recall@K, MRR, rang median, scalare cu galeria, regăsire vs. generare |
| Baseline generativ | `i2t/generative.py` | BLIP (titluri generate) |
| Feedback | `i2t/feedback.py` | voturi 👍/👎 salvate în SQLite |
| CLI | `i2t/cli.py` | `serve`, `add`, `search`, `caption`, `eval` |
| Date | `scripts/prepare_coco_subset.py` | descarcă un subset COCO pentru evaluare |
| Teste | `tests/` | 41 de teste (index, colecții, evaluare, API), fără descărcarea modelelor |

Documentație detaliată: [docs/ARHITECTURA.md](docs/ARHITECTURA.md) și [docs/EXPERIMENTE.md](docs/EXPERIMENTE.md).

## Instalare

Necesită Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Modelele se descarcă automat de pe Hugging Face la prima folosire (B/32 ≈ 600 MB, B/16 ≈ 600 MB,
multilingv ≈ 1 GB în total, BLIP ≈ 1 GB).

## Pornire

```bash
python -m i2t.cli serve          # http://127.0.0.1:8000
```

La prima pornire se creează două colecții demo (`demo-en`, `demo-ro`, câte 71 de descrieri) în `./storage`.

Interfața are patru taburi:
- **Căutare**: încarci o imagine (drag-and-drop, selectare sau lipire), alegi colecția și unul sau mai multe
  modele (coloanele se afișează alăturat pentru comparație), primești titlul, alternativele, scorurile și timpii;
  poți vota fiecare rezultat și poți adăuga un titlu generat de BLIP.
- **Colecții**: creezi colecții, adaugi texte manual sau din `.txt`/`.csv`/`.jsonl`, filtrezi și ștergi.
- **Evaluare**: afișează raportul din `results/eval.json` (tabele, curba de scalare) și statisticile voturilor.
- **Despre**: explicația metodei și limitările.

## Linia de comandă

```bash
python -m i2t.cli caption poza.jpg --collection demo-en            # titlu + alternative
python -m i2t.cli caption poza.jpg --generative                    # + titlu generat BLIP
python -m i2t.cli search  poza.jpg --model clip-b16 --top-k 10 --json
python -m i2t.cli --model clip-multilingual caption poza.jpg --collection demo-ro
python -m i2t.cli add --collection recenzii --texts comentarii.csv # creează/actualizează o colecție
```

## API (rezumat)

| Metodă | Cale | Descriere |
|---|---|---|
| GET | `/api/models` | modelele disponibile și starea lor |
| GET/POST/DELETE | `/api/collections[/{nume}]` | listare, creare, ștergere colecții |
| GET/POST | `/api/collections/{nume}/texts` | pagini de texte (cu filtru) / adăugare |
| POST | `/api/collections/{nume}/upload` | încărcare `.txt/.csv/.jsonl` |
| DELETE | `/api/collections/{nume}/texts/{id}` | ștergere text |
| POST | `/api/search` | imagine → top-K texte (multipart: `image`, `collection`, `model`, `top_k`, `min_score`) |
| POST | `/api/caption` | ca mai sus + `title`, `alternatives`, opțional `generative=true` |
| POST | `/api/feedback`, GET `/api/feedback/stats` | voturi de relevanță |
| GET | `/api/evaluation` | ultimul raport de evaluare |

```bash
curl -F image=@poza.jpg -F collection=demo-en -F model=clip-b32 -F top_k=3 localhost:8000/api/caption
```

## Evaluare

```bash
python scripts/prepare_coco_subset.py --n 1000          # ≈ 240 MB adnotări + imagini
python -m i2t.cli eval --pairs data/coco/pairs.csv --images data/coco/images \
       --models clip-b32,clip-b16,clip-multilingual --sizes 100,250,500,1000 --generative
```

Raportul (`results/eval.json`) apare în tab-ul **Evaluare**. Rezultatele obținute sunt în
[docs/EXPERIMENTE.md](docs/EXPERIMENTE.md).

## Teste

```bash
python -m pytest -q
```

Testele folosesc un codificator fals (vectori după culoare), deci rulează în secunde și fără rețea.

## Docker

```bash
docker build -t i2t .
docker run -p 8000:8000 -v i2t-storage:/app/storage -v i2t-hf:/app/.hf i2t
```

## Limitări

- Sistemul **alege** dintre textele existente; nu generează text nou (BLIP este oferit doar ca termen de comparație).
- Calitatea titlului depinde de acoperirea colecției; folosește `min_score` pentru a evita titluri înșelătoare.
- Modelele B/32 și B/16 sunt antrenate în principal pe engleză; pentru română folosește `clip-multilingual`.
- Căutarea este exactă și liniară în numărul de texte (rapidă până la sute de mii); pentru milioane de texte
  ar trebui înlocuită cu un index aproximativ (FAISS, HNSW).
- Modelele pot reproduce prejudecăți din datele de antrenare; nu sunt potrivite pentru decizii asupra persoanelor.
