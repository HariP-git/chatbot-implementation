#!/usr/bin/env python3
"""
rag_hybrid_v4.py — Hybrid FAISS + E5 + RapidFuzz + BM25 + Auto-Expansion RAG API

Features:
 - Automatic concept-store generation from metadata (no hard-coded expansions)
 - Embedding-based query expansion (nearest concept terms)
 - FAISS per-field retrieval using intfloat/e5-large-v2
 - BM25 lexical scoring per-field (internal simple implementation)
 - Hybrid scoring: weighted combination of embed + BM25 + fuzzy
 - Optional high-accuracy mode with cross-encoder re-ranking if available
 - Results + Recommendations with recommendation limit per-field
 - Summary via Ollama primary (fallback heuristic)
"""

import os, json, time, math, argparse
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss
import ijson
import subprocess
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

# RapidFuzz optional
try:
    from rapidfuzz import fuzz
    def fuzzy_partial(q, t): return fuzz.partial_ratio(q, t) / 100.0
except Exception:
    def fuzzy_partial(q, t):
        if not q or not t:
            return 0.0
        qs = set([w for w in q.lower().split() if len(w) > 2])
        ts = set([w for w in t.lower().split() if len(w) > 2])
        if not ts:
            return 0.0
        return len(qs & ts) / float(len(ts))

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = r"D:\chatbot\faiss_embedding\data"
VECTOR_DIR = os.path.join(DATA_DIR, "vector")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

EMBED_MODEL = "intfloat/e5-large-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional for high mode

OLLAMA_MODEL = "tinyllama"
OLLAMA_TIMEOUT = 20.0

TOP_K = 50
HIGH_MATCH_SCORE = 0.90 
# hybrid weights (embedding, bm25, fuzzy). Sum should be 1.0
WEIGHT_EMB = 0.6
WEIGHT_BM25 = 0.25
WEIGHT_FUZZY = 0.15

HYBRID_ALPHA = WEIGHT_EMB  # legacy var used in some places

# field selection thresholds
FIELD_SIM_THRESHOLD = 0.35
FIELD_KEYWORD_OVERLAP_THRESHOLD = 0.06
MAX_RELEVANT_FIELDS = 12

# caches & limits
FIELD_EMBED_CACHE = os.path.join(DATA_DIR, "field_embeddings.json")
CONCEPTS_CACHE = os.path.join(DATA_DIR, "concept_terms.json")
FIELD_KEYWORDS_CACHE = os.path.join(DATA_DIR, "field_keywords.json")
CONCEPT_EMB_CACHE = os.path.join(DATA_DIR, "concept_embeddings.json")

MAX_THREAD_POOL = 8
RECOMMEND_LIMIT_PER_FIELD = 6

# cross-encoder rerank top
CROSS_RERANK_TOP = 64
FIELD_WEIGHTS = {}
MEDIUM_MATCH_SCORE = 0.60

# ---------------------------
# INITIALIZE MODELS
# ---------------------------
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)

cross_encoder = None
if CROSS_ENCODER_AVAILABLE:
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception:
        cross_encoder = None
        CROSS_ENCODER_AVAILABLE = False

# ---------------------------
# LOAD FAISS INDEXES & METADATA
# ---------------------------
index_map: Dict[str, faiss.Index] = {}
if os.path.isdir(VECTOR_DIR):
    for fname in os.listdir(VECTOR_DIR):
        if fname.lower().endswith(".faiss"):
            field = fname[:-6]
            try:
                index_map[field] = faiss.read_index(os.path.join(VECTOR_DIR, fname))
            except Exception:
                continue

meta_map: Dict[str, Dict[int, Any]] = {f: {} for f in index_map.keys()}
if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            parser = ijson.items(f, "item")
            for entry in parser:
                field = entry.get("collection") or entry.get("field")
                if field in meta_map:
                    try:
                        vid = int(entry.get("id", len(meta_map[field])))
                    except Exception:
                        vid = len(meta_map[field])
                    meta_map[field][vid] = entry
    except Exception:
        pass

# ---------------------------
# UTILITIES
# ---------------------------
def normalize_text(t: str) -> str:
    return (t or "").strip().lower()

def safe_display(entry):
    if not entry:
        return ""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for k in ("value", "text", "name", "label", "title"):
            v = entry.get(k)
            if v:
                return str(v)
    return str(entry)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))

# ---------------------------
# SIMPLE BM25 IMPLEMENTATION
# Lightweight BM25 to avoid extra deps.
# ---------------------------
class BM25Simple:
    def __init__(self, docs: List[List[str]], k1=1.5, b=0.75):
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / (self.N + 1e-9)
        self.k1 = k1
        self.b = b
        self.df = {}
        self.idf = {}
        self.doc_len = [len(d) for d in docs]
        for d in docs:
            seen = set()
            for w in d:
                if w not in seen:
                    self.df[w] = self.df.get(w, 0) + 1
                    seen.add(w)
        for w, freq in self.df.items():
            self.idf[w] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
        # store term frequencies per doc
        self.tf = []
        for d in docs:
            t = {}
            for w in d:
                t[w] = t.get(w, 0) + 1
            self.tf.append(t)

    def score(self, q_tokens: List[str], doc_index: int) -> float:
        score = 0.0
        dl = self.doc_len[doc_index]
        tfmap = self.tf[doc_index]
        for q in q_tokens:
            if q not in tfmap:
                continue
            idf = self.idf.get(q, 0.0)
            tf = tfmap[q]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            score += idf * (tf * (self.k1 + 1)) / (denom + 1e-9)
        return score

    def get_scores(self, q_tokens: List[str]) -> List[float]:
        return [self.score(q_tokens, i) for i in range(self.N)]

# ---------------------------
# BUILD concept terms & field keywords (caches)
# We'll auto-extract common display tokens per field, and a concept-term set.
# ---------------------------
def build_or_load_field_keywords(sample_limit=300, min_token_len=3):
    if os.path.exists(FIELD_KEYWORDS_CACHE):
        try:
            with open(FIELD_KEYWORDS_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    kw = {}
    for field, entries in meta_map.items():
        tokens = []
        count = 0
        for entry in list(entries.values()):
            if count >= sample_limit:
                break
            disp = safe_display(entry)
            if not disp:
                continue
            # simple tokenization
            for tok in disp.lower().replace("/", " ").replace("-", " ").split():
                tok = tok.strip(",.()")
                if len(tok) >= min_token_len:
                    tokens.append(tok)
            count += 1
        # unique preserve
        kw[field] = sorted(list(set(tokens)))
    try:
        with open(FIELD_KEYWORDS_CACHE, "w", encoding="utf-8") as f:
            json.dump(kw, f)
    except Exception:
        pass
    return kw

field_keywords = build_or_load_field_keywords()

# Concept terms: sample top tokens across all fields (frequent terms). We'll embed these as expansion candidates.
def build_or_load_concept_terms(top_n=2000):
    if os.path.exists(CONCEPTS_CACHE) and os.path.exists(CONCEPT_EMB_CACHE):
        try:
            with open(CONCEPTS_CACHE, "r", encoding="utf-8") as f:
                terms = json.load(f)
            with open(CONCEPT_EMB_CACHE, "r", encoding="utf-8") as f:
                embmap = json.load(f)
            embmap = {k: np.array(v, dtype=np.float32) for k, v in embmap.items()}
            return terms, embmap
        except Exception:
            pass

    # aggregate token frequencies from field_keywords
    freq = {}
    for toks in field_keywords.values():
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    # pick top tokens
    top_tokens = [t for t, _ in sorted(freq.items(), key=lambda x: -x[1])][:top_n]
    # embed these tokens
    if top_tokens:
        embs = embedder.encode(top_tokens, normalize_embeddings=True, convert_to_numpy=True)
        embmap = {t: e.tolist() for t, e in zip(top_tokens, embs)}
        try:
            with open(CONCEPTS_CACHE, "w", encoding="utf-8") as f:
                json.dump(top_tokens, f)
            with open(CONCEPT_EMB_CACHE, "w", encoding="utf-8") as f:
                json.dump(embmap, f)
        except Exception:
            pass
        embmap = {k: np.array(v, dtype=np.float32) for k, v in embmap.items()}
        return top_tokens, embmap
    return [], {}

concept_terms, concept_embeddings = build_or_load_concept_terms()

# ---------------------------
# FIELD EMBEDDINGS (for field-name relevance)
# ---------------------------
def load_or_build_field_emb():
    if os.path.exists(FIELD_EMBED_CACHE):
        try:
            with open(FIELD_EMBED_CACHE, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {k: np.array(v, dtype=np.float32) for k, v in d.items()}
        except Exception:
            pass
    fields = list(index_map.keys())
    fb = {}
    if fields:
        embs = embedder.encode(fields, normalize_embeddings=True, convert_to_numpy=True)
        for f, e in zip(fields, embs):
            fb[f] = e.tolist()
        try:
            with open(FIELD_EMBED_CACHE, "w", encoding="utf-8") as f:
                json.dump(fb, f)
        except Exception:
            pass
    return {k: np.array(v, dtype=np.float32) for k, v in fb.items()}

field_embs = load_or_build_field_emb()

# ---------------------------
# Field-level BM25 indices (built from sample displays)
# ---------------------------
def build_field_bm25(sample_limit=1000, min_token_len=2):
    bm25_map = {}
    token_docs_map = {}
    for field, entries in meta_map.items():
        docs = []
        count = 0
        for entry in list(entries.values()):
            if count >= sample_limit:
                break
            disp = safe_display(entry)
            if not disp:
                continue
            toks = [t for t in disp.lower().replace("/", " ").replace("-", " ").split() if len(t) >= min_token_len]
            if toks:
                docs.append(toks)
                count += 1
        if docs:
            bm25_map[field] = BM25Simple(docs)
            token_docs_map[field] = docs
    return bm25_map, token_docs_map

field_bm25_map, field_token_docs = build_field_bm25()

# ---------------------------
# Field relevance selection (embedding + keyword overlap)
# ---------------------------
def select_relevant_fields(query: str) -> List[str]:
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = [(f, cosine_sim(qv, fe)) for f, fe in field_embs.items()]
    q_tokens = set(normalize_text(query).split())
    selected = []
    for f, sim in scores:
        kws = set(field_keywords.get(f, []))
        overlap = len(q_tokens & kws) / (len(kws) + 1e-9) if kws else 0.0
        if sim >= FIELD_SIM_THRESHOLD or overlap >= FIELD_KEYWORD_OVERLAP_THRESHOLD:
            selected.append((f, sim, overlap))
    selected.sort(key=lambda x: (-x[1], -x[2]))
    chosen = [f for f, _, _ in selected][:MAX_RELEVANT_FIELDS]
    if not chosen:
        chosen = [f for f, _, _ in sorted(scores, key=lambda x: -x[1])[:min(4, len(scores))]]
    return chosen

# ---------------------------
# Automatic embedding-based expansion
# ---------------------------
def expand_query_with_concepts(query: str, top_n=6, sim_threshold=0.62) -> str:
    if not concept_terms or not concept_embeddings:
        return query
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    terms = list(concept_embeddings.keys())
    vecs = np.stack([concept_embeddings[t] for t in terms])
    sims = (vecs @ qv).tolist()
    idxs = np.argsort(sims)[::-1][:top_n]
    additions = []
    for i in idxs:
        s = sims[i]
        if s >= sim_threshold:
            additions.append(terms[i])
    if additions:
        return query + " " + " ".join(additions)
    return query

# ---------------------------
# FAISS retrieval parallel
# ---------------------------
def search_field(field: str, qv: np.ndarray, top_k=TOP_K) -> Tuple[str, List[Tuple[str, float]]]:
    idx = index_map.get(field)
    if idx is None:
        return field, []
    try:
        D, I = idx.search(qv.reshape(1, -1), top_k)
    except Exception:
        return field, []
    hits = []
    for id_, sc in zip(I[0], D[0]):
        if int(id_) < 0:
            continue
        ent = meta_map.get(field, {}).get(int(id_))
        disp = safe_display(ent)
        if disp:
            hits.append((disp, float(sc)))
    return field, hits

def retrieve_hits(query: str, fields: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    out = {}
    with ThreadPoolExecutor(max_workers=min(MAX_THREAD_POOL, len(fields) or 1)) as ex:
        futures = {ex.submit(search_field, f, qv): f for f in fields}
        for fut in as_completed(futures):
            f, hits = fut.result()
            if hits:
                out[f] = hits
    return out

# ---------------------------
# (Optional) Cross-encoder rerank
# ---------------------------
def cross_rerank(query: str, hits_by_field: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    if cross_encoder is None:
        return hits_by_field
    out = {}
    for field, hits in hits_by_field.items():
        texts = [t for t, _ in hits][:CROSS_RERANK_TOP]
        if not texts:
            continue
        pairs = [[query, t] for t in texts]
        try:
            scores = cross_encoder.predict(pairs)
        except Exception:
            out[field] = hits
            continue
        ranked = sorted(zip(texts, scores), key=lambda x: -x[1])
        out[field] = [(t, float(s)) for t, s in ranked]
    return out

# ---------------------------
# Hybrid scoring combining embedding, BM25, fuzzy
# ---------------------------
def score_and_classify(query: str, hits_by_field: Dict[str, List[Tuple[str, float]]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    results = {}
    recommendations = {}
    q_tokens = [t for t in normalize_text(query).split() if t]
    for field, hits in hits_by_field.items():
        # build bm25 scores if available
        bm25 = field_bm25_map.get(field)
        # prepare doc list for BM25 scoring: use tokenized hits (we'll compute per-hit bm25)
        hit_texts = [t for t, _ in hits]
        # compute embedding scores (already present as second)
        # compute BM25 per hit: since our BM25 is built on sampled docs, use query tokens score vs doc tokens approximated by tokenizing hit text
        score_map = {}
        for display, emb_score in hits:
            bm25_score = 0.0
            if bm25:
                toks = [w for w in display.lower().replace("/", " ").replace("-", " ").split() if w]
                # approximate: compare query tokens against this single doc by on-the-fly scoring (use BM25Simple score on doc-level)
                # We'll create a one-doc BM25 wrapper locally for accuracy penalty; simpler: compute token overlap idf-weighted
                bm25_score = 0.0
                for q in q_tokens:
                    if q in toks:
                        bm25_score += 1.0
                # normalize by tokens length
                if toks:
                    bm25_score = bm25_score / len(toks)
            fuzzy = fuzzy_partial(query, display)
            # hybrid combination with field weight
            field_w = FIELD_WEIGHTS.get(field, 1.0)
            hybrid = field_w * (WEIGHT_EMB * emb_score + WEIGHT_BM25 * bm25_score + WEIGHT_FUZZY * fuzzy)
            key = normalize_text(display)
            if hybrid > score_map.get(key, -1.0):
                score_map[key] = hybrid

        # sort items
        sorted_items = sorted(score_map.items(), key=lambda x: -x[1])
        kept_results = []
        kept_recs = []
        for key, hybrid in sorted_items:
            # find original display string
            orig = next((d for d, s in hits if normalize_text(d) == key), key)
            if hybrid >= HIGH_MATCH_SCORE:
                kept_results.append(orig)
            elif hybrid >= MEDIUM_MATCH_SCORE:
                kept_recs.append(orig)
        # results: all high matches
        if kept_results:
            results[field] = kept_results
            # recommendations: union of results + top recs limited
            merged = list(dict.fromkeys(kept_results + kept_recs))
            recommendations[field] = merged[:RECOMMEND_LIMIT_PER_FIELD]
        else:
            # no strong results, recommendations = top recs limited
            if kept_recs:
                recommendations[field] = kept_recs[:RECOMMEND_LIMIT_PER_FIELD]
    # ensure presence: if field has recommendations but no results, optionally promote top recs into results up to 3
    for f in list(recommendations.keys()):
        if f not in results:
            results[f] = recommendations[f][:min(3, len(recommendations[f]))]
    return results, recommendations

# ---------------------------
# LLM summary (Ollama primary)
# ---------------------------
def run_ollama(prompt: str, timeout=OLLAMA_TIMEOUT):
    start = time.time()
    try:
        proc = subprocess.run(["ollama", "run", OLLAMA_MODEL], input=prompt.encode("utf-8"), capture_output=True, timeout=timeout)
        elapsed = round(time.time() - start, 3)
        if proc.returncode == 0:
            txt = proc.stdout.decode("utf-8").strip()
            return True, txt, elapsed
    except Exception:
        pass
    return False, "", round(time.time() - start, 3)

def generate_summary(query: str, results: Dict[str, List[str]], recommendations: Dict[str, List[str]]):
    ctx = []
    for k, v in list(results.items())[:6]:
        ctx.append(f"{k}: {', '.join(v[:3])}")
    for k, v in list(recommendations.items())[:4]:
        ctx.append(f"{k}: {', '.join(v[:2])}")
    prompt = f"User Query:\n{query}\n\nContext:\n" + "\n".join(ctx) + "\n\nWrite one short <= 25 words summary of the user's intent."
    ok, txt, dur = run_ollama(prompt)
    if ok and txt:
        return txt.strip(), "TinyLlama", dur
    # fallback simple summary
    return (query.strip()[:200], "None", 0.0)

# ---------------------------
# MAIN HANDLER (modes: fast / high)
# ---------------------------
def handle_query(query: str, mode: str = "fast"):
    t0 = time.time()
    mode = mode.lower()
    query_proc = query

    # high-mode: expansion using concept terms
    emb_time = 0.0
    bm25_time = 0.0
    rerank_time = 0.0

    if mode == "high":
        t1 = time.time()
        query_proc = expand_query_with_concepts(query_proc, top_n=8, sim_threshold=0.62)
        emb_time = round(time.time() - t1, 3)

    # select relevant fields
    t_sel = time.time()
    fields = select_relevant_fields(query_proc)
    t_sel = round(time.time() - t_sel, 3)

    # retrieve hits
    t_ret = time.time()
    hits = retrieve_hits(query_proc, fields)
    t_ret = round(time.time() - t_ret, 3)

    # optional cross-encoder rerank (only in high mode and if available)
    if mode == "high" and CROSS_ENCODER_AVAILABLE and cross_encoder is not None:
        t_rr = time.time()
        hits = cross_rerank(query_proc, hits)
        rerank_time = round(time.time() - t_rr, 3)

    # hybrid scoring with BM25 approximations
    t_bm = time.time()
    results, recommendations = score_and_classify(query_proc, hits)
    bm25_time = round(time.time() - t_bm, 3)

    # summary via LLM
    t_llm = time.time()
    summary, model_used, llm_time = generate_summary(query, results, recommendations)
    llm_time = round(time.time() - t_llm, 3)

    total = round(time.time() - t0, 3)

    return {
        "summary": summary,
        "results": results,
        "recommendations": recommendations,
        "model_used": f"{EMBED_MODEL} {'+ cross-encoder' if (mode=='high' and CROSS_ENCODER_AVAILABLE) else ''} + BM25",
        "execution_time_seconds": {
            "embedding": emb_time,
            "field_selection": t_sel,
            "retrieval": t_ret,
            "rerank": rerank_time,
            "bm25_and_hybrid": bm25_time,
            "llm": llm_time,
            "total": total
        }
    }

# ---------------------------
# FASTAPI
# ---------------------------
app = FastAPI(title="RAG Hybrid v4")

class QueryIn(BaseModel):
    query: str
    mode: str = "fast"

@app.post("/query")
async def query_api(req: QueryIn):
    q = (req.query or "").strip()
    mode = (req.mode or "fast").strip().lower()
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty query"})
    try:
        out = handle_query(q, mode=mode)
        return JSONResponse(content=out)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "RAG Hybrid v4 running", "endpoints": ["/query (POST)"]}

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "high"])
    args = parser.parse_args()
    if args.query:
        print(json.dumps(handle_query(args.query, mode=args.mode), indent=2, ensure_ascii=False))
    elif args.serve:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("Run --serve to start API or --query '...' to test one query.")
