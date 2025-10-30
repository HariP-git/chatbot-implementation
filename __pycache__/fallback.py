# ============================================================
# fallback.py — Optimized FAISS + TinyLlama + Gemini RAG
# Cleaned Repetition + Hallucination-Free Output
# ============================================================

import os, json, time, faiss, requests, subprocess, numpy as np, torch, argparse, ijson
from datetime import datetime
from typing import Dict, Any, Tuple, List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"D:\chatbot\faiss_embedding\data"
VECTOR_DIR = os.path.join(DATA_DIR, "vector")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
GEMINI_SAVE_PATH = os.path.join(DATA_DIR, "geminidata.json")

EMBED_MODEL = "intfloat/e5-large-v2"
OLLAMA_MODEL = "tinyllama"
OLLAMA_TIMEOUT = 18
GEMINI_API_KEY = "AIzaSyDxS9k2_wgcYa_Dl2Qlee4_cG2Vo4vhpZ4"
GEMINI_TIMEOUT = 18

FIELDS = [
    "Subindustry","account_category","cd_companyMaximumAge","cd_geographyCountries",
    "cd_sicCode","company_status","company_type","employee_range","hiring_ind",
    "includes_c","includes_p","job_description","job_function","job_title_level",
    "location_type","marketable_flag_c","marketable_flag_p","post_code",
    "sic_code_description","supressionType","turnover_range","town_county_country",
    "technologies"
]

# ============================================================
# UTILITIES
# ============================================================
def log(msg: str): print(f"{datetime.now().isoformat()} - {msg}")

def clean_list(values: List[str]) -> List[str]:
    """Deduplicate and normalize results."""
    seen = set()
    cleaned = []
    for v in values:
        if not v: continue
        norm = v.strip().lower().replace("_", " ").replace("-", " ")
        if norm not in seen:
            seen.add(norm)
            cleaned.append(v.strip())
    return cleaned

# ============================================================
# LOAD FAISS INDEXES
# ============================================================
log(f"🔹 Loading FAISS indexes from {VECTOR_DIR}")
index_map = {}
for fname in os.listdir(VECTOR_DIR):
    if fname.endswith(".faiss"):
        field = fname[:-6]
        try:
            idx = faiss.read_index(os.path.join(VECTOR_DIR, fname))
            index_map[field] = idx
            log(f"✅ Loaded {field} ({idx.ntotal} vectors)")
        except Exception as e:
            log(f"⚠️ Failed to load {field}: {e}")

if not index_map:
    raise RuntimeError("❌ No FAISS indexes found.")

# ============================================================
# LOAD METADATA (STREAM MODE)
# ============================================================
log(f"🔹 Loading metadata from {META_PATH} (stream mode)")
meta_map = {f: {} for f in FIELDS}
count = 0
with open(META_PATH, "r", encoding="utf-8") as f:
    parser = ijson.items(f, "item")
    for entry in parser:
        field = entry.get("collection") or entry.get("field")
        if field in meta_map:
            vid = entry.get("id", len(meta_map[field]))
            meta_map[field][int(vid)] = entry
            count += 1
            if count % 100000 == 0:
                log(f"   ...loaded {count:,} metadata entries")
log(f"✅ Metadata loaded for {len(meta_map)} fields (total {count:,} entries)")

# ============================================================
# EMBEDDING MODEL
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)
log(f"🧠 Embedding model loaded on {device}")

# ============================================================
# MODEL HELPERS
# ============================================================
def run_ollama(prompt: str) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(["ollama", "run", OLLAMA_MODEL],
                              input=prompt.encode("utf-8"),
                              capture_output=True,
                              timeout=OLLAMA_TIMEOUT)
        if proc.returncode == 0:
            return True, proc.stdout.decode().strip()
    except subprocess.TimeoutExpired:
        log("⚠️ Ollama timed out.")
    except Exception as e:
        log(f"⚠️ Ollama error: {e}")
    return False, ""

def run_gemini(prompt: str) -> Tuple[bool, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-bison-001:generate?key={GEMINI_API_KEY}"
    body = {"prompt": {"text": prompt}, "temperature": 0.3, "maxOutputTokens": 400}
    try:
        r = requests.post(url, json=body, timeout=GEMINI_TIMEOUT)
        data = r.json()
        text = data.get("candidates", [{}])[0].get("output", "") or \
               data.get("candidates", [{}])[0].get("content", {}).get("text", "")
        if text:
            record = {"timestamp": datetime.now().isoformat(), "prompt": prompt, "response": text}
            with open(GEMINI_SAVE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True, text.strip()
    except Exception as e:
        log(f"⚠️ Gemini error: {e}")
    return False, ""

# ============================================================
# RAG PIPELINE
# ============================================================
def retrieve_field_results(query: str, field: str, top_k=10):
    if field not in index_map: return []
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    try:
        scores, ids = index_map[field].search(qv, top_k)
    except Exception as e:
        log(f"⚠️ FAISS search error {field}: {e}")
        return []
    results = []
    for sc, vid in zip(scores[0], ids[0]):
        entry = meta_map.get(field, {}).get(int(vid))
        if entry and entry.get("value"):
            results.append((entry["value"], float(sc)))
    return results

def build_inclusion_exclusion(results: Dict[str, list]) -> Dict[str, Any]:
    structured = {}
    for f, vals in results.items():
        if not vals:
            structured[f] = {"inclusion": None, "exclusion": None}
            continue
        vals_sorted = sorted(vals, key=lambda x: -x[1])
        inclusion = clean_list([v for v, _ in vals_sorted[:5]])
        exclusion = clean_list([v for v, _ in vals_sorted[-5:]])
        structured[f] = {"inclusion": inclusion or None, "exclusion": exclusion or None}
    return structured

# ============================================================
# REASONING
# ============================================================
def rag_reasoning(query: str, structured: Dict[str, Any]) -> Tuple[str, str, float]:
    ctx_parts = []
    for k, v in structured.items():
        incl = v.get("inclusion") or []
        if incl:
            ctx_parts.append(f"{k}: {', '.join(incl[:3])}")
    context = "\n".join(ctx_parts[:12])  # limit context to prevent repetition
    prompt = f"""
User Query:
{query}

Context (cleaned and de-duplicated facts):
{context}

Write 2 short, factual sentences describing what the user wants.
Then write one actionable recommendation related to these inclusions.
Avoid repetition.
"""
    start = time.time()
    ok, text = run_ollama(prompt)
    model_used = "TinyLlama"
    if not ok or not text.strip():
        ok2, text2 = run_gemini(prompt)
        if ok2:
            text = text2
            model_used = "Gemini"
        else:
            text = "Unable to summarize query."
            model_used = "None"
    return text.strip(), model_used, round(time.time() - start, 2)

# ============================================================
# MAIN HANDLER
# ============================================================
def handle_query(query: str) -> Dict[str, Any]:
    t0 = time.time()
    results = {f: retrieve_field_results(query, f) for f in FIELDS}
    structured = build_inclusion_exclusion(results)
    summary, model_used, t_llm = rag_reasoning(query, structured)
    top_incl = [v["inclusion"][0] for v in structured.values() if v["inclusion"]]
    rec = f"You may explore opportunities related to {', '.join(clean_list(top_incl[:10]))}."
    return {
        "query": query,
        "summary": summary,
        "recommendations": rec,
        "results": structured,
        "model_used": model_used,
        "execution_time_seconds": {"llm": t_llm, "total": round(time.time() - t0, 2)},
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="FAISS + TinyLlama + Gemini RAG")

class QueryIn(BaseModel):
    query: str

@app.post("/query")
async def query_api(req: QueryIn):
    return JSONResponse(content=handle_query(req.query))

@app.get("/")
def root():
    return {"message": "RAG API running", "docs": "/docs"}

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.query:
        print(json.dumps(handle_query(args.query), indent=2, ensure_ascii=False))
    elif args.serve:
        log("🚀 Starting FAISS+RAG API on http://127.0.0.1:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
