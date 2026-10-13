# Arhitectură și decizii de proiectare

## 1. Problema

Pentru o imagine de interogare se cere lista celor mai apropiate semantic texte dintr-o colecție
(descrieri, titluri, comentarii). Cea mai importantă aplicație este **titrarea**: primul text devine titlul,
următoarele sunt alternative.

Dificultatea centrală este *semantic gap*: pixelii nu spun ce înfățișează imaginea. Soluția folosită este
un spațiu vectorial comun pentru imagini și text (CLIP), în care apropierea geometrică înseamnă apropiere de sens.

## 2. Fluxul de date

```
              OFFLINE / la adăugarea textelor                     ONLINE / la fiecare interogare
   ┌────────────┐   ┌───────────────┐   ┌──────────────────┐     ┌─────────┐   ┌─────────────────┐
   │ texte (UI/ │──►│ codificator   │──►│ emb_<model>.npy  │     │ imagine │──►│ codificator     │
   │ fișiere)   │   │ de text       │   │ (n × d, L2-norm) │     │ încărcată│  │ de imagini      │
   └────────────┘   └───────────────┘   └────────┬─────────┘     └─────────┘   └────────┬────────┘
                                                 │ E·qᵀ (cosinus)                       │ q (d)
                                                 └───────────────► scoruri ◄────────────┘
                                                                      │ top-K + softmax
                                                                      ▼
                                                      titlu = textul de rang 1, restul = alternative
```

1. **Reprezentare vizuală.** Imaginea este redimensionată/normalizată de procesorul modelului și trecută prin
   codificatorul de imagini; rezultatul este un vector de 512 dimensiuni, normalizat L2.
2. **Reprezentări textuale precalculate.** Fiecare text din colecție este codificat o singură dată și salvat.
   Vectorii sunt păstrați **separat pentru fiecare model**, deoarece spațiile modelelor nu sunt compatibile.
3. **Comparare.** Cu vectori normalizați, similaritatea cosinus este produsul scalar, deci scorurile pentru
   toată colecția se obțin cu un singur produs matrice–vector (`E @ q`).
4. **Clasament.** Se iau cele mai mari K scoruri (`argpartition`, apoi sortare doar a celor K).
   Scorurile sunt transformate și în probabilități cu `softmax(logit_scale · scor)`, unde `logit_scale`
   (≈100) este temperatura învățată de CLIP; probabilitatea măsoară „cât de clar câștigă” primul rezultat
   față de restul colecției.

## 3. Structura codului

```
i2t/
  encoders/
    base.py          interfața Encoder (encode_texts / encode_images, vectori L2-normalizați)
    hf_clip.py       CLIP din Hugging Face (B/32, B/16); compatibil transformers 4.x și 5.x
    multilingual.py  imagini: CLIP B/32; text: model multilingv distilat în același spațiu
    __init__.py      ModelRegistry: creare leneșă + cache, protejat cu lock
  index.py           TextIndex: căutare exactă, încărcare de texte (.txt/.csv/.jsonl)
  store.py           colecții persistente, indexare incrementală, ștergere cu menținerea alinierii
  service.py         RetrievalService: validare imagine, search/caption, timpi pe etape
  feedback.py        voturi în SQLite
  generative.py      BLIP (încărcare leneșă)
  evaluation.py      metrici și rularea comparațiilor
  api.py             FastAPI + servirea interfeței
  cli.py             comenzi
  web/               index.html, style.css, app.js (fără build, fără dependențe externe)
```

Separarea pe straturi (encoder → index → store → service → api/cli) permite testarea fiecărui strat izolat:
testele înlocuiesc modelul cu un codificator fals (vectori după culoare), deci verifică logica reală de
clasament, persistență și API fără a descărca modele.

## 4. Decizii

| Decizie | Motiv | Alternativa respinsă |
|---|---|---|
| Vectori normalizați + produs scalar | un singur `matmul`, scoruri comparabile, independente de magnitudine | distanța euclidiană (aceeași ordonare pe vectori normalizați, dar scoruri mai puțin interpretabile) |
| Căutare exactă (NumPy) | pentru zeci/sute de mii de texte durează sub câteva zeci de ms; fără aproximare | FAISS/HNSW: necesare abia la milioane de texte (vezi „Extensii”) |
| Vectori pe disc per (colecție, model) | adăugarea de texte codifică doar textele noi; schimbarea modelului nu invalidează celelalte | un index unic reconstruit la fiecare modificare |
| Încărcare leneșă a modelelor | pornire rapidă; modelele neutilizate nu consumă memorie | încărcare la pornire |
| Modele multilingve ca model separat | imaginile rămân în spațiul CLIP B/32, doar textul se schimbă; permite română fără reantrenare | traducerea automată a interogărilor |
| UI fără framework | nicio etapă de build, fără dependențe de rețea la rulare, ușor de auditat | React/Vite |
| Construirea DOM-ului cu `textContent` | textele din colecții sunt date ale utilizatorului: nu se interpretează ca HTML (fără XSS) | `innerHTML` cu escapare manuală |
| Validare imagini (10 MB, 50 Mpx, decodare reală) | previne fișiere false și bombe de decompresie | verificarea doar a extensiei |
| Scrieri atomice (`tmp` + `replace`) pentru fișierele de colecție | un crash nu lasă JSON-uri trunchiate | scriere directă |

## 5. Siguranță și robustețe

- **Nume de colecții** validate printr-o expresie regulată (`[A-Za-z0-9_-]`, max. 64); acest lucru exclude
  traversarea de directoare (`../`).
- **Texte**: normalizarea spațiilor, ignorarea duplicatelor și a textelor de peste 500 de caractere.
- **Încărcări**: tipuri acceptate verificate, limită de 5 MB pentru fișiere text, 10 MB pentru imagini.
- **Concurență**: `CollectionStore` folosește un `RLock`, `ModelRegistry` un `Lock`; endpoint-urile sunt
  sincrone și rulează în threadpool-ul FastAPI.
- **Intimitate**: imaginile încărcate nu se salvează pe disc; feedback-ul păstrează doar un hash scurt
  al imaginii, nu imaginea.
- **Autentificare**: lipsește intenționat (aplicație locală). Pentru expunere publică ar trebui adăugate
  autentificare, limitarea ratei și limite de dimensiune la nivel de proxy.

## 6. Metodologia evaluării

Set de date: subset COCO val2017 (imagini + 5 descrieri/imagine). Galeria conține **toate** descrierile din set.

- **Recall@K (imagine→text):** o interogare reușește dacă *oricare* dintre cele ~5 descrieri proprii apare în
  primele K rezultate.
- **MRR și rang median:** calculate pe poziția primei descrieri corecte.
- **Text→imagine:** direcția inversă (fiecare descriere caută imaginea corectă printre toate imaginile).
- **Scalare cu galeria:** R@1 când galeria conține doar primele n imagini (și descrierile lor); arată cum
  scad performanțele când cresc distractorii.
- **Regăsire vs. generare (open-set):** pentru fiecare imagine, galeria *exclude* propriile descrieri
  (situația reală: colecția nu conține textul perfect). Titlul regăsit se compară cu cel generat de BLIP prin
  (a) cosinus imagine–titlu și (b) cel mai mare cosinus text–text față de descrierile de referință.
  Ambele măsuri folosesc embeddings CLIP, deci favorizează ușor metodele aliniate cu CLIP; interpretarea
  trebuie făcută cu această rezervă.
- Departajarea egalităților este conservatoare: rangul unui rezultat = 1 + numărul celor *strict* mai bune.

## 7. Extensii posibile

- Index aproximativ (FAISS IVF/HNSW) pentru colecții de milioane de texte.
- Reordonare (*re-ranking*) a primelor K rezultate cu un model cross-modal (de ex. BLIP-ITM).
- Adaptare fină (*fine-tuning*) pe perechi din domeniul propriu (medical, e-commerce).
- Folosirea voturilor din feedback pentru calibrare sau pentru antrenare cu *hard negatives*.
- Interogări combinate (imagine + text) prin media ponderată a vectorilor.
