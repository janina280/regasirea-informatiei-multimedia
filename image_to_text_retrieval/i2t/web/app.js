"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/** Creează un element DOM; textul se pune mereu prin textContent (fără risc XSS). */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.setAttribute("style", v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `Eroare ${r.status}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 2200);
}

/* ------------------------------------------------------------ taburi */
$$(".tab").forEach(b => b.addEventListener("click", () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "collections") loadCollections();
  if (b.dataset.tab === "eval") loadEval();
}));

/* ------------------------------------------------------------ căutare */
const state = { file: null, models: [], collections: [] };

function setStatus(msg, err = false, busy = false) {
  const s = $("#status");
  s.className = "status" + (err ? " err" : "");
  s.replaceChildren(busy ? h("span", { class: "spinner" }) : "", msg);
}

async function initSearch() {
  const [models, cols] = await Promise.all([api("/api/models"), api("/api/collections")]);
  state.models = models; state.collections = cols;
  const box = $("#models");
  box.replaceChildren(...models.map(m =>
    h("label", { class: "model-opt" },
      h("input", { type: "checkbox", value: m.key, checked: m.default }),
      h("span", {}, m.label, h("small", {}, `${m.languages} · ${m.description}`)))));
  fillCollectionSelect();
}

function fillCollectionSelect() {
  const sel = $("#collection");
  const prev = sel.value;
  sel.replaceChildren(...state.collections.map(c => h("option", { value: c.name }, `${c.name} (${c.count})`)));
  if (state.collections.some(c => c.name === prev)) sel.value = prev;
}

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) { toast("Alege un fișier imagine."); return; }
  state.file = f;
  const img = $("#preview");
  img.src = URL.createObjectURL(f);
  img.hidden = false;
  $("#drop-hint").hidden = true;
  $("#go").disabled = false;
}

const drop = $("#drop");
drop.addEventListener("click", () => $("#file").click());
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
$("#file").addEventListener("change", e => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
document.addEventListener("paste", e => {
  const f = [...(e.clipboardData?.files || [])].find(x => x.type.startsWith("image/"));
  if (f) setFile(f);
});
$("#topk").addEventListener("input", e => ($("#topk-val").textContent = e.target.value));

async function runCaption(model, withGenerative) {
  const fd = new FormData();
  fd.append("image", state.file);
  fd.append("collection", $("#collection").value);
  fd.append("model", model);
  fd.append("top_k", $("#topk").value);
  const ms = $("#minscore").value;
  if (ms !== "") fd.append("min_score", ms);
  fd.append("generative", withGenerative ? "true" : "false");
  return api("/api/caption", { method: "POST", body: fd });
}

$("#go").addEventListener("click", async () => {
  const models = $$("#models input:checked").map(i => i.value);
  if (!models.length) { setStatus("Bifează cel puțin un model.", true); return; }
  if (!$("#collection").value) { setStatus("Creează mai întâi o colecție de texte.", true); return; }
  const wantGen = $("#generative").checked;
  $("#go").disabled = true;
  setStatus(wantGen ? "Se calculează (BLIP și modelele noi se încarcă la prima folosire, poate dura)…" : "Se calculează (la prima folosire a unui model, încărcarea poate dura)…", false, true);
  try {
    const out = await Promise.all(models.map((m, i) => runCaption(m, wantGen && i === 0)));
    renderResults(out);
    setStatus("");
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("#go").disabled = false;
  }
});

function modelLabel(key) { return state.models.find(m => m.key === key)?.label || key; }

function renderResults(outs) {
  const first = outs[0];
  const root = $("#results");
  root.replaceChildren();

  const titleCard = h("div", { class: "card title-card" },
    h("div", { class: "label" }, "Titlu descriptiv · " + modelLabel(first.model)),
    first.title
      ? h("div", { class: "big" }, first.title)
      : h("div", { class: "big muted" }, "Niciun text nu depășește pragul"),
    first.title && h("button", { class: "secondary", onclick: () => navigator.clipboard.writeText(first.title).then(() => toast("Titlu copiat")) }, "Copiază titlul"),
    first.alternatives.length > 0 && h("div", { class: "alts-wrap" }, h("div", { class: "label", style: "margin-top:10px" }, "Alternative"),
      h("ul", { class: "alts" }, first.alternatives.map(a => h("li", {}, a)))),
    first.generative && h("div", { class: "gen-box" },
      h("div", { class: "label" }, "Titlu generat (BLIP, nu din colecție)"),
      first.generative.error
        ? h("div", { class: "muted" }, first.generative.error)
        : h("div", { class: "big2" }, first.generative.title,
            h("span", { class: "muted" }, `  · ${first.generative.time_ms} ms`))));
  root.append(titleCard);

  const cols = outs.map(o => h("div", { class: "card model-col" },
    h("h3", {}, modelLabel(o.model), h("small", {}, `${o.corpus_size} texte`)),
    o.results.length === 0 ? h("p", { class: "muted" }, "Fără rezultate.") : o.results.map(r => hit(o, r)),
    h("div", { class: "timing" },
      `sincronizare index ${o.timing_ms.index_sync} ms · codificare imagine ${o.timing_ms.image_encode} ms · căutare ${o.timing_ms.search} ms`)));
  root.append(h("div", { class: "cols-grid" }, cols));
}

function hit(o, r) {
  const pct = Math.max(0, Math.min(100, ((r.score - 0.1) / 0.3) * 100)); // 0.10..0.40 -> 0..100%
  const vote = (rel) => async (e) => {
    const box = e.currentTarget.parentElement;
    try {
      await api("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection: o.collection, model: o.model, image_hash: o.image_hash,
          text: r.text, rank: r.rank, score: r.score, relevant: rel }) });
      $$("button", box).forEach(b => b.classList.remove("on"));
      e.currentTarget.classList.add("on");
    } catch (err) { toast(err.message); }
  };
  return h("div", { class: "hit" },
    h("div", { class: "txt" }, h("span", { class: "rank" }, r.rank + "."), r.text),
    h("div", { class: "vote" }, h("button", { title: "Relevant", onclick: vote(true) }, "👍"), " ",
      h("button", { title: "Irelevant", onclick: vote(false) }, "👎")),
    h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })),
    h("div", { class: "meta" }, h("span", {}, `cosinus ${r.score.toFixed(3)}`), h("span", {}, `${(r.probability * 100).toFixed(1)}%`)));
}

/* ------------------------------------------------------------ colecții */
const colState = { name: null, offset: 0, limit: 25, total: 0, q: "" };

async function loadCollections() {
  await loadCollectionsList();
  if (colState.name && !state.collections.some(c => c.name === colState.name)) await selectCollection(null);
}

function colMsg(msg, err = false) { const e = $("#col-msg"); e.textContent = msg; e.className = "status" + (err ? " err" : ""); }

async function selectCollection(name) {
  colState.name = name; colState.offset = 0; colState.q = "";
  $("#col-filter").value = "";
  $("#col-title").textContent = name || "Selectează o colecție";
  $("#col-body").hidden = !name;
  $("#col-delete").hidden = !name;
  await loadCollectionsList();
  if (name) await loadTexts();
}
async function loadCollectionsList() {
  state.collections = await api("/api/collections");
  fillCollectionSelect();
  $("#col-list").replaceChildren(...state.collections.map(c =>
    h("li", { class: c.name === colState.name ? "sel" : "", onclick: () => selectCollection(c.name) },
      h("span", {}, c.name), h("span", { class: "n" }, c.count))));
}

async function loadTexts() {
  const p = new URLSearchParams({ offset: colState.offset, limit: colState.limit, q: colState.q });
  const data = await api(`/api/collections/${encodeURIComponent(colState.name)}/texts?${p}`);
  colState.total = data.total;
  const tb = $("#col-table tbody");
  tb.replaceChildren(...data.items.map(it => h("tr", {},
    h("td", {}, it.text),
    h("td", { class: "act" }, h("button", { title: "Șterge", onclick: async () => {
      await api(`/api/collections/${encodeURIComponent(colState.name)}/texts/${it.id}`, { method: "DELETE" });
      await loadTexts(); await loadCollectionsList();
    } }, "✕")))));
  const end = Math.min(colState.offset + colState.limit, data.total);
  $("#pageinfo").textContent = data.total ? `${colState.offset + 1}–${end} din ${data.total}` : "gol";
  $("#prev").disabled = colState.offset === 0;
  $("#next").disabled = end >= data.total;
}

$("#prev").addEventListener("click", () => { colState.offset = Math.max(0, colState.offset - colState.limit); loadTexts(); });
$("#next").addEventListener("click", () => { colState.offset += colState.limit; loadTexts(); });
let filterTimer;
$("#col-filter").addEventListener("input", e => {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(() => { colState.q = e.target.value; colState.offset = 0; loadTexts(); }, 250);
});

$("#col-new").addEventListener("submit", async e => {
  e.preventDefault();
  const name = $("#col-name").value.trim();
  try {
    await api("/api/collections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    $("#col-name").value = ""; colMsg("");
    await selectCollection(name);
  } catch (err) { colMsg(err.message, true); }
});

$("#col-add").addEventListener("click", async () => {
  const texts = $("#col-text").value.split("\n").map(s => s.trim()).filter(Boolean);
  if (!texts.length) return;
  try {
    const r = await api(`/api/collections/${encodeURIComponent(colState.name)}/texts`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ texts }) });
    $("#col-text").value = "";
    toast(`+${r.added} texte (${r.skipped} ignorate)`);
    await loadTexts(); await loadCollectionsList();
  } catch (err) { toast(err.message); }
});

$("#col-file").addEventListener("change", async e => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await api(`/api/collections/${encodeURIComponent(colState.name)}/upload`, { method: "POST", body: fd });
    toast(`+${r.added} texte (${r.skipped} ignorate)`);
    await loadTexts(); await loadCollectionsList();
  } catch (err) { toast(err.message); }
});

$("#col-delete").addEventListener("click", async () => {
  if (!confirm(`Ștergi colecția „${colState.name}” împreună cu toți vectorii ei?`)) return;
  await api(`/api/collections/${encodeURIComponent(colState.name)}`, { method: "DELETE" });
  await selectCollection(null);
});

/* ------------------------------------------------------------ evaluare */
const PALETTE = ["#0e7c86", "#e8a33d", "#c8553d", "#5b8e7d", "#6b7a8f"];

async function loadEval() {
  const root = $("#eval-body");
  const [ev, fb] = await Promise.all([
    api("/api/evaluation").catch(e => ({ error: e.message })),
    api("/api/feedback/stats").catch(() => null),
  ]);
  root.replaceChildren();
  if (ev.error) {
    root.append(h("div", { class: "card" }, h("h2", {}, "Evaluare"),
      h("p", { class: "muted" }, ev.error),
      h("p", {}, "Exemplu: ", h("code", {}, "python scripts/prepare_coco_subset.py && python -m i2t.cli eval --pairs data/coco/pairs.csv --images data/coco/images"))));
  } else {
    root.append(h("div", { class: "eval-grid" },
      metricCard(ev, "image_to_text", "Imagine → text"),
      metricCard(ev, "text_to_image", "Text → imagine")));
    if (ev.models.some(m => m.gallery_scaling)) root.append(h("div", { class: "card", style: "margin-top:16px" }, scalingChart(ev)));
    if (ev.retrieval_vs_generation) root.append(h("div", { class: "card", style: "margin-top:16px" }, genCard(ev.retrieval_vs_generation)));
    root.append(h("p", { class: "muted" }, `Set de date: ${ev.dataset.n_images} imagini, ${ev.dataset.n_texts} descrieri · ${ev.created}`));
  }
  if (fb && fb.total > 0) {
    root.append(h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Feedback utilizatori"),
      h("p", { class: "muted" }, `${fb.total} voturi, ${(fb.relevant / fb.total * 100).toFixed(0)}% relevante`),
      h("table", { class: "metric-table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", {}, "Voturi"), h("th", {}, "Relevant"))),
        h("tbody", {}, fb.by_model.map(m => h("tr", {}, h("td", {}, m.model), h("td", {}, m.votes), h("td", {}, (m.relevant_rate * 100).toFixed(0) + "%")))))));
  }
}

function metricCard(ev, dir, title) {
  const cols = ["R@1", "R@5", "R@10", "MRR", "median_rank"];
  const best = Object.fromEntries(cols.map(c => [c, Math[c === "median_rank" ? "min" : "max"](...ev.models.map(m => m[dir][c]))]));
  return h("div", { class: "card" }, h("h2", {}, title),
    h("table", { class: "metric-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Model"), cols.map(c => h("th", {}, c === "median_rank" ? "Rang med." : c)))),
      h("tbody", {}, ev.models.map(m => h("tr", {}, h("td", {}, modelLabel(m.model)),
        cols.map(c => {
          const v = m[dir][c];
          const txt = c === "median_rank" ? v.toFixed(0) : (v * 100).toFixed(1) + (c === "MRR" ? "" : "%");
          return h("td", { style: v === best[c] ? "font-weight:700;color:var(--accent)" : "" }, c === "MRR" ? v.toFixed(3) : txt);
        }))))),
    h("p", { class: "muted", style: "margin:8px 0 0" }, "R@K = proporția interogărilor cu un rezultat corect în primele K."));
}

function scalingChart(ev) {
  const W = 640, H = 260, L = 44, R = 12, T = 12, B = 34;
  const series = ev.models.filter(m => m.gallery_scaling?.length);
  const xs = series[0].gallery_scaling.map(p => p.n_images);
  const xmax = Math.max(...xs), xmin = Math.min(...xs);
  const X = v => L + ((v - xmin) / Math.max(xmax - xmin, 1)) * (W - L - R);
  const Y = v => T + (1 - v) * (H - T - B);
  const NS = "http://www.w3.org/2000/svg";
  const s = (tag, a = {}, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt != null) e.textContent = txt; return e; };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "R@1 în funcție de mărimea galeriei" });
  for (let g = 0; g <= 1.0001; g += 0.25) {
    svg.append(s("line", { x1: L, x2: W - R, y1: Y(g), y2: Y(g), stroke: "currentColor", "stroke-opacity": .12 }));
    svg.append(s("text", { x: L - 6, y: Y(g) + 4, "text-anchor": "end" }, (g * 100).toFixed(0) + "%"));
  }
  xs.forEach(x => svg.append(s("text", { x: X(x), y: H - 12, "text-anchor": "middle" }, x)));
  series.forEach((m, i) => {
    const c = PALETTE[i % PALETTE.length];
    const pts = m.gallery_scaling.map(p => [X(p.n_images), Y(p["R@1"])]);
    svg.append(s("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: c, "stroke-width": 2.5 }));
    pts.forEach(p => svg.append(s("circle", { cx: p[0], cy: p[1], r: 3.5, fill: c })));
  });
  return h("div", {}, h("h2", {}, "R@1 în funcție de mărimea galeriei (imagine → text)"), svg,
    h("div", { class: "legend" }, series.map((m, i) => h("span", {}, h("i", { style: `background:${PALETTE[i % PALETTE.length]}` }), modelLabel(m.model)))),
    h("p", { class: "muted" }, "Axa X: numărul de imagini din galerie. Cu cât galeria are mai multe texte, cu atât crește numărul de distractori."));
}

function genCard(g) {
  const row = (name, v) => h("tr", {}, h("td", {}, name), h("td", {}, v.image_text_sim.toFixed(3)), h("td", {}, v.ref_sim.toFixed(3)));
  return h("div", {}, h("h2", {}, "Regăsire vs. generare (scenariu open-set)"),
    h("p", { class: "muted" }, `${g.n_images} imagini; galeria nu conține propriile descrieri ale imaginii. Encoder de scor: ${modelLabel(g.encoder)}.`),
    h("table", { class: "metric-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Metodă"), h("th", {}, "Cosinus imagine–titlu"), h("th", {}, "Similaritate cu referința"))),
      h("tbody", {}, row("Regăsire (din colecție)", g.retrieval_open_set), row("Generare (BLIP)", g.generation_blip))),
    h("h3", {}, "Exemple"),
    g.examples.map(e => h("div", { class: "ex" }, h("div", {}, h("b", {}, "Imagine: "), e.image),
      h("div", {}, h("b", {}, "Referință: "), e.reference),
      h("div", {}, h("b", {}, "Regăsit: "), e.retrieved),
      h("div", {}, h("b", {}, "Generat: "), e.generated))));
}

/* ------------------------------------------------------------ start */
initSearch().catch(e => setStatus("Nu pot contacta serverul: " + e.message, true));
