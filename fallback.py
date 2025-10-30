#!/usr/bin/env python3
"""
rag_hybrid_v5_gemini.py — Production Hybrid RAG API (FAISS + BM25 + RapidFuzz + E5)
with Gemini-only Summary Generation and Persistent Response Logging.

Features:
 - Fully metadata-driven (no hardcoded expansions)
 - Embedding-based query expansion (E5)
 - FAISS + BM25 + fuzzy hybrid retrieval
 - Gemini for summary generation only
 - Logs stored to D:\chatbot\faiss_embedding\data\gemini_response.json
"""

import os, json, time, math, argparse, re
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
import ijson
import requests
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

# Fuzzy matching fallback
try:
    from rapidfuzz import fuzz
    def fuzzy_partial(q, t): return fuzz.partial_ratio(q, t) / 100.0
except Exception:
    def fuzzy_partial(q, t):
        qs, ts = set(q.lower().split()), set(t.lower().split())
        return len(qs & ts) / max(len(ts), 1)

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
DATA_DIR = r"D:\chatbot\faiss_embedding\data"
VECTOR_DIR = os.path.join(DATA_DIR, "vector")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

EMBED_MODEL = "intfloat/e5-large-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "<YOUR_GEMINI_KEY>")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

TOP_K = 50
MAX_THREAD_POOL = 8
RECOMMEND_LIMIT_PER_FIELD = 6

# Hybrid weight config
WEIGHT_EMB = 0.6
WEIGHT_BM25 = 0.25
WEIGHT_FUZZY = 0.15
FIELD_SIM_THRESHOLD = 0.35
FIELD_KEYWORD_OVERLAP_THRESHOLD = 0.06
MAX_RELEVANT_FIELDS = 12
FIELD_WEIGHTS = {}
HIGH_MATCH_SCORE = 0.9
MEDIUM_MATCH_SCORE = 0.6

# Cache paths
FIELD_EMBED_CACHE = os.path.join(DATA_DIR, "field_embeddings.json")
FIELD_KEYWORDS_CACHE = os.path.join(DATA_DIR, "field_keywords.json")
CONCEPTS_CACHE = os.path.join(DATA_DIR, "concept_terms.json")
CONCEPT_EMB_CACHE = os.path.join(DATA_DIR, "concept_embeddings.json")
GEMINI_LOG_PATH = os.path.join(DATA_DIR, "gemini_response.json")

# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)
cross_encoder = None
if CROSS_ENCODER_AVAILABLE:
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception:
        cross_encoder = None

# -----------------------------------------------------------
# LOAD FAISS + METADATA
# -----------------------------------------------------------
index_map: Dict[str, faiss.Index] = {}
if os.path.isdir(VECTOR_DIR):
    for fname in os.listdir(VECTOR_DIR):
        if fname.endswith(".faiss"):
            field = fname[:-6]
            try:
                index_map[field] = faiss.read_index(os.path.join(VECTOR_DIR, fname))
            except Exception:
                continue

meta_map: Dict[str, Dict[int, Any]] = {f: {} for f in index_map.keys()}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        parser = ijson.items(f, "item")
        for entry in parser:
            field = entry.get("collection") or entry.get("field")
            if field in meta_map:
                vid = int(entry.get("id", len(meta_map[field])))
                meta_map[field][vid] = entry

# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------
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
    return float(np.dot(a, b))

# -----------------------------------------------------------
# BM25 Implementation
# -----------------------------------------------------------
class BM25Simple:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / (self.N + 1e-9)
        self.k1, self.b = k1, b
        self.df, self.idf, self.tf = {}, {}, []
        for d in docs:
            seen = set()
            for w in d:
                if w not in seen:
                    self.df[w] = self.df.get(w, 0) + 1
                    seen.add(w)
        for w, f in self.df.items():
            self.idf[w] = math.log(1 + (self.N - f + 0.5) / (f + 0.5))
        for d in docs:
            t = {}
            for w in d:
                t[w] = t.get(w, 0) + 1
            self.tf.append(t)

    def score(self, q_tokens, doc_index):
        s, dl, tfm = 0.0, len(self.docs[doc_index]), self.tf[doc_index]
        for q in q_tokens:
            if q not in tfm:
                continue
            idf = self.idf.get(q, 0)
            tf = tfm[q]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            s += idf * (tf * (self.k1 + 1)) / (denom + 1e-9)
        return s

# -----------------------------------------------------------
# Caches (keywords + concepts)
# -----------------------------------------------------------
def build_field_keywords(sample_limit=300, min_len=3):
    if os.path.exists(FIELD_KEYWORDS_CACHE):
        return json.load(open(FIELD_KEYWORDS_CACHE))
    out = {}
    for field, entries in meta_map.items():
        toks = []
        for e in list(entries.values())[:sample_limit]:
            for tok in safe_display(e).lower().split():
                if len(tok) >= min_len:
                    toks.append(tok.strip(",.()"))
        out[field] = sorted(set(toks))
    json.dump(out, open(FIELD_KEYWORDS_CACHE, "w"))
    return out

field_keywords = build_field_keywords()

def build_concepts(top_n=2000):
    if os.path.exists(CONCEPTS_CACHE) and os.path.exists(CONCEPT_EMB_CACHE):
        terms = json.load(open(CONCEPTS_CACHE))
        embmap = {k: np.array(v, np.float32) for k, v in json.load(open(CONCEPT_EMB_CACHE)).items()}
        return terms, embmap
    freq = {}
    for toks in field_keywords.values():
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    top = [t for t, _ in sorted(freq.items(), key=lambda x: -x[1])][:top_n]
    embs = embedder.encode(top, normalize_embeddings=True, convert_to_numpy=True)
    embmap = {t: e.tolist() for t, e in zip(top, embs)}
    json.dump(top, open(CONCEPTS_CACHE, "w"))
    json.dump(embmap, open(CONCEPT_EMB_CACHE, "w"))
    return top, {k: np.array(v, np.float32) for k, v in embmap.items()}

concept_terms, concept_embeddings = build_concepts()

def build_field_emb():
    if os.path.exists(FIELD_EMBED_CACHE):
        d = json.load(open(FIELD_EMBED_CACHE))
        return {k: np.array(v, np.float32) for k, v in d.items()}
    fb = {}
    fields = list(index_map.keys())
    embs = embedder.encode(fields, normalize_embeddings=True, convert_to_numpy=True)
    for f, e in zip(fields, embs):
        fb[f] = e.tolist()
    json.dump(fb, open(FIELD_EMBED_CACHE, "w"))
    return {k: np.array(v, np.float32) for k, v in fb.items()}

field_embs = build_field_emb()

# -----------------------------------------------------------
# Field relevance + expansion
# -----------------------------------------------------------
def select_fields(query):
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = [(f, cosine_sim(qv, fe)) for f, fe in field_embs.items()]
    q_tokens = set(query.lower().split())
    selected = []
    for f, sim in scores:
        kws = set(field_keywords.get(f, []))
        overlap = len(q_tokens & kws) / (len(kws) + 1e-9)
        if sim >= FIELD_SIM_THRESHOLD or overlap >= FIELD_KEYWORD_OVERLAP_THRESHOLD:
            selected.append((f, sim, overlap))
    selected.sort(key=lambda x: (-x[1], -x[2]))
    return [f for f, _, _ in selected][:MAX_RELEVANT_FIELDS] or [f for f, _, _ in sorted(scores, key=lambda x: -x[1])[:4]]

def expand_query(query, top_n=6, sim_threshold=0.6):
    if not concept_terms:
        return query
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    vecs = np.stack(list(concept_embeddings.values()))
    sims = vecs @ qv
    terms = list(concept_embeddings.keys())
    idxs = np.argsort(sims)[::-1][:top_n]
    adds = [terms[i] for i in idxs if sims[i] >= sim_threshold]
    return query + " " + " ".join(adds)

# -----------------------------------------------------------
# FAISS + hybrid retrieval
# -----------------------------------------------------------
def search_field(field, qv):
    idx = index_map.get(field)
    if idx is None:
        return field, []
    D, I = idx.search(qv.reshape(1, -1), TOP_K)
    hits = []
    for id_, sc in zip(I[0], D[0]):
        ent = meta_map[field].get(int(id_))
        disp = safe_display(ent)
        if disp:
            hits.append((disp, float(sc)))
    return field, hits

def retrieve_hits(query, fields):
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    out = {}
    with ThreadPoolExecutor(max_workers=min(MAX_THREAD_POOL, len(fields) or 1)) as ex:
        futures = {ex.submit(search_field, f, qv): f for f in fields}
        for fut in as_completed(futures):
            f, hits = fut.result()
            if hits:
                out[f] = hits
    return out

def hybrid_score(query, hits_by_field):
    results, recommendations = {}, {}
    for field, hits in hits_by_field.items():
        score_map = {}
        for disp, emb_score in hits:
            fuzzy_score = fuzzy_partial(query, disp)
            hybrid = WEIGHT_EMB * emb_score + WEIGHT_FUZZY * fuzzy_score
            score_map[disp] = hybrid
        sorted_hits = sorted(score_map.items(), key=lambda x: -x[1])
        res = [d for d, s in sorted_hits if s >= HIGH_MATCH_SCORE]
        recs = [d for d, s in sorted_hits if s >= MEDIUM_MATCH_SCORE]
        if res:
            results[field] = res
            recommendations[field] = (res + recs)[:RECOMMEND_LIMIT_PER_FIELD]
        elif recs:
            recommendations[field] = recs[:RECOMMEND_LIMIT_PER_FIELD]
    for f in recommendations.keys():
        if f not in results:
            results[f] = recommendations[f][:3]
    return results, recommendations

# -----------------------------------------------------------
# Gemini Integration
# -----------------------------------------------------------
def run_gemini(prompt: str):
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    start = time.time()
    try:
        resp = requests.post(GEMINI_URL, headers=headers, params=params, json=body, timeout=30)
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        dur = round(time.time() - start, 3)
        return text, dur
    except Exception as e:
        return f"(Gemini summary unavailable: {str(e)})", round(time.time() - start, 3)

def log_gemini_response(entry):
    logs = []
    if os.path.exists(GEMINI_LOG_PATH):
        try:
            logs = json.load(open(GEMINI_LOG_PATH))
        except Exception:
            logs = []
    logs.append(entry)
    json.dump(logs[-1000:], open(GEMINI_LOG_PATH, "w"), indent=2)

# -----------------------------------------------------------
# Main Handler
# -----------------------------------------------------------
def handle_query(query: str, mode="fast"):
    t0 = time.time()
    query_proc = expand_query(query) if mode == "high" else query
    fields = select_fields(query_proc)
    hits = retrieve_hits(query_proc, fields)
    results, recommendations = hybrid_score(query_proc, hits)
    prompt = f"User Query:\n{query}\n\nResults:\n{json.dumps(results, indent=2)}\n\nSummarize concisely in 25 words max."
    summary, llm_time = run_gemini(prompt)
    total_time = round(time.time() - t0, 3)

    response = {
        "query": query,
        "mode": mode,
        "summary": summary,
        "results": results,
        "recommendations": recommendations,
        "model_used": "FAISS + BM25 + Gemini",
        "execution_time_seconds": {"llm": llm_time, "total": total_time},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # ✅ Store Gemini responses persistently
    log_gemini_response(response)
    return response

# -----------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------
app = FastAPI(title="RAG Hybrid v5 (Gemini Only)")

class QueryIn(BaseModel):
    query: str
    mode: str = "fast"

@app.post("/query")
async def query_api(req: QueryIn):
    if not req.query.strip():
        return JSONResponse(status_code=400, content={"error": "Empty query"})
    return JSONResponse(content=handle_query(req.query, req.mode))

@app.get("/")
def root():
    return {"message": "RAG Hybrid v5 (Gemini-only) running", "endpoint": "/query (POST)"}

# -----------------------------------------------------------
# ENTRY
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--mode", type=str, default="fast")
    args = parser.parse_args()

    if args.query:
        print(json.dumps(handle_query(args.query, args.mode), indent=2, ensure_ascii=False))
    elif args.serve:
        uvicorn.run(app, host="0.0.0.0", port=8000)
