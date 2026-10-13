# Experimente și rezultate

Toate numerele provin din `results/eval.json`, generat de:

```bash
python scripts/prepare_coco_subset.py --n 1000
python -m i2t.cli eval --pairs data/coco/pairs.csv --images data/coco/images \
       --models clip-b32,clip-b16,clip-multilingual --sizes 100,250,500,1000 --generative --gen-images 100
```

**Set de date:** 1000 de imagini COCO val2017 alese aleatoriu (seed 0), cu 5002 de descrieri în engleză
(galeria de texte). **Hardware:** laptop Apple, accelerare MPS; o singură rulare (fără repetări, deci fără
intervale de încredere).

## 1. Compararea modelelor (galerie completă: 1000 imagini / 5002 texte)

| Model | Direcție | R@1 | R@5 | R@10 | MRR | Rang median |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | imagine→text | 0,697 | 0,895 | 0,938 | 0,784 | 1 |
| CLIP ViT-B/32 | text→imagine | 0,496 | 0,784 | 0,880 | 0,625 | 2 |
| CLIP ViT-B/16 | imagine→text | **0,723** | **0,919** | **0,956** | **0,807** | 1 |
| CLIP ViT-B/16 | text→imagine | **0,524** | **0,803** | **0,893** | **0,648** | 1 |
| CLIP multilingv | imagine→text | 0,634 | 0,862 | 0,931 | 0,735 | 1 |
| CLIP multilingv | text→imagine | 0,436 | 0,750 | 0,866 | 0,578 | 2 |

Cost de calcul (ms, medie): codificare imagine B/32 13,0 · B/16 29,2 · multilingv 13,9; codificare text ≈ 1,2–1,4.

**Observații**
- **B/16 > B/32** pe toate metricile (+2,6 puncte procentuale R@1 imagine→text), cu un cost de ≈2,2× la
  codificarea imaginii. Căutarea propriu-zisă rămâne neglijabilă (sub 1 ms pe 5002 de texte).
- **Modelul multilingv este mai slab pe engleză** decât B/32 (−6,3 pp R@1), deși folosește același
  codificator de imagini: codificatorul de text este un model *distilat*, care aproximează spațiul CLIP,
  iar aproximarea costă puțină precizie. Compensația este suportul pentru română și alte limbi (vezi §4).
- **Direcția imagine→text este mai ușoară** decât text→imagine la toate modelele, pentru că aici o interogare
  reușește dacă *oricare* din cele ~5 descrieri corecte este recuperată, deci există mai multe ținte valide.

## 2. Scalarea cu mărimea galeriei (R@1, imagine→text)

| Imagini în galerie | B/32 | B/16 | multilingv |
|---|---|---|---|
| 100 | 0,910 | 0,980 | 0,970 |
| 250 | 0,832 | 0,872 | 0,824 |
| 500 | 0,774 | 0,800 | 0,752 |
| 1000 | 0,697 | 0,723 | 0,634 |

Performanța scade monoton când cresc distractorii (≈ −21 pp pentru B/32 între 100 și 1000 de imagini):
o colecție mare și densă de texte similare face titrarea mai greu de nimerit. Prin urmare, R@1 măsurat pe
o colecție mică nu se extrapolează la una mare. (Diferența dintre modele la 100 de imagini este mai
mică decât zgomotul probabil: 100 de interogări înseamnă ±3–4 pp.)

## 3. Regăsire vs. generare (100 de imagini, scenariu open-set)

Galeria *nu* conține descrierile imaginii interogate (situația realistă: textul perfect nu există în colecție).

| Metodă | Cosinus imagine–titlu | Similaritate cu referința |
|---|---|---|
| Regăsire (B/32, din restul descrierilor) | **0,310** | 0,810 |
| Generare (BLIP-base) | 0,295 | **0,868** |

**Interpretare.** Titlurile generate sunt mai apropiate de descrierile umane de referință (0,868 vs 0,810):
BLIP produce text *adaptat* imaginii, în timp ce regăsirea trebuie să se mulțumească cu cel mai bun text deja
existent, care poate menționa obiecte greșite. Regăsirea câștigă la cosinus imagine–titlu, dar această măsură
este **circulară**: textul regăsit este ales tocmai maximizând acest cosinus în același spațiu CLIP, deci avantajul
nu trebuie citit ca superioritate calitativă. Concluzia prudentă este că *într-o colecție generică, titrarea
generativă potrivește mai bine descrierile umane; regăsirea rămâne preferabilă când textele trebuie să provină
dintr-un set controlat* (titluri editoriale, etichete de catalog, texte verificate, fără risc de halucinație).

Exemple (din `results/eval.json`):

| Referință | Regăsit | Generat (BLIP) |
|---|---|---|
| A young lad swings his Wilson tennis racquet. | A child holds a racket with a hat on his head. | a young boy playing tennis on a tennis court |
| Two young kids interacting in the grocery store produce section. | A young girl speaking on a cell phone near a produce vendor. | people shopping at a market |
| A table with a napkin, doughnut on a plate, and a cup of coffee. | A small doughnut inside a cup that's sitting on a table. | a cup of coffee next to a dont |

Al treilea exemplu arată un defect al generării (greșeala „dont”), iar primul un defect al regăsirii
(descrierea găsită nu menționează tenisul).

## 4. Demonstrația multilingvă (calitativ, o imagine)

Imaginea cu două pisici pe o canapea roz, colecția `demo-ro` (texte în română):

| Model | Rang 1 | Cosinus |
|---|---|---|
| CLIP ViT-B/32 (doar engleză) | „un tablou cu un peisaj într-un muzeu de artă” ✗ | 0,229 |
| CLIP multilingv | „doi pisici care se odihnesc pe o canapea roz” ✓ | 0,328 |

Modelul englez nu înțelege textul românesc (scorurile sunt apropiate între ele și de valori de zgomot),
iar cel multilingv regăsește corect. Aceasta este o singură ilustrare, nu o evaluare; o evaluare riguroasă ar
cere un set de date cu descrieri românești (de exemplu COCO tradus).

## 5. Amenințări la validitate

- O singură rulare, o singură selecție de 1000 de imagini; nu s-au calculat intervale de încredere.
- COCO conține scene comune; rezultatele nu se transferă automat la domenii specializate (medical, artă).
- Descrierile COCO din galerie sunt scurte și generice; într-o colecție reală, textele (titluri, comentarii) sunt mai
  eterogene, iar R@K ar putea diferi mult.
- Măsurile din §3 folosesc CLIP B/32 și sunt, prin construcție, favorabile metodelor aliniate cu CLIP.
- Voturile din interfață (tab-ul Evaluare) nu sunt incluse aici; sunt subiective și nu provin dintr-un protocol controlat.
- Modelele CLIP au prejudecăți documentate; setul COCO nu a fost analizat pentru echitate.

## 6. Pași următori propuși pentru lucrarea de semestru

1. Repetarea cu mai multe selecții aleatoare (seed-uri) și raportarea mediei ± abaterea standard.
2. Evaluare pe un set cu descrieri românești (traducere + verificare) pentru a cuantifica modelul multilingv.
3. Reordonare a primelor K rezultate cu un model cross-modal și măsurarea câștigului în R@1.
4. Adaptare fină pe un domeniu ales și comparație înainte/după.
5. Index aproximativ (FAISS/HNSW) pe 1M de texte: compromis viteză–acuratețe.
