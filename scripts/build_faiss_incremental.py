# ============================================================
# ⚡ build_faiss_incremental_gpu_fieldwise.py
# 🚀 GPU + Field-Aware Incremental FAISS Builder (Final Version)
# ============================================================
import os
import re
import json
import faiss
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
DATA_DIR = r"D:\chatbot\faiss_embedding\data"
VECTOR_DIR = os.path.join(DATA_DIR, "vector")  # ✅ each FAISS file will be stored here
META_PATH = os.path.join(DATA_DIR, "metadata.json")
STATE_PATH = os.path.join(DATA_DIR, "index_state.json")

# 🧠 Strong context-aware embedding model
EMBED_MODEL = "intfloat/e5-large-v2"

BATCH_SIZE = 128
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# List of JSON files (without .json extension)
COLLECTIONS = [
    "Subindustry",
    "account_category",
    "cd_companyMaximumAge",
    "cd_geographyCountries",
    "cd_sicCode",
    "company_status",
    "company_type",
    "employee_range",
    "hiring_ind",
    "includes_c",
    "includes_p",
    "job_description",
    "job_function",
    "job_title_level",
    "location_type",
    "marketable_flag_c",
    "marketable_flag_p",
    "post_code",
    "sic_code_description",
    "supressionType",
    "turnover_range",
    "town_county_country",
    "technologies",
]

# ============================================================
# 🧩 HELPER FUNCTIONS
# ============================================================
def normalize_value(val: str) -> str:
    """Clean and normalize text/numeric content."""
    if not val:
        return ""
    val = str(val).strip().replace(",", "").replace("£", "").replace("$", "")

    # Handle numeric ranges like 250K–500K → between 250000 and 500000
    m = re.match(r"(\d+(?:\.\d+)?)([kKmM]?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)([kKmM]?)", val)
    if m:
        a, asuf, b, bsuf = m.groups()

        def conv(x, s):
            x = float(x)
            return x * 1_000 if s.lower() == "k" else x * 1_000_000 if s.lower() == "m" else x

        lo, hi = conv(a, asuf), conv(b, bsuf)
        return f"between {int(lo)} and {int(hi)}"

    # Expand abbreviations
    replacements = {"uk": "United Kingdom", "usa": "United States", "us": "United States"}
    for k, v in replacements.items():
        val = re.sub(rf"\b{k}\b", v, val, flags=re.IGNORECASE)

    return val


def prepare_field_text(field: str, value: str) -> str:
    """Attach field context to value for richer embeddings."""
    return f"Field: {field}. Value: {normalize_value(value)}"


def embed_texts(model, texts, batch_size, device):
    """Batch embed with GPU/CPU and progress display."""
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, device=device)
        all_embs.append(embs)
    return np.vstack(all_embs).astype(np.float32)


# ============================================================
# 🚀 MAIN FUNCTION
# ============================================================
def build_fieldwise_index():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    print(f"🧠 Loading embedding model: {EMBED_MODEL}")
    print(f"💻 Using device: {DEVICE.upper()} {'(GPU 🚀)' if USE_GPU else '(CPU)'}")

    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    torch.set_num_threads(os.cpu_count())

    # Load previous state and metadata
    index_state = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            index_state = json.load(f)

    metadata = []
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Process each file
    for collection in COLLECTIONS:
        json_path = os.path.join(DATA_DIR, f"{collection}.json")
        faiss_path = os.path.join(VECTOR_DIR, f"{collection}.faiss")

        if not os.path.exists(json_path):
            print(f"⚠️ {collection}.json not found. Skipping.")
            continue

        last_modified = os.path.getmtime(json_path)
        last_indexed = index_state.get(collection, {}).get("last_indexed", 0)
        if last_modified <= last_indexed:
            print(f"⏩ Skipping {collection} (no changes since last index).")
            continue

        print(f"\n📂 Indexing collection: {collection}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {collection}.json: {e}")
            continue

        if not isinstance(data, list) or not data:
            print(f"⚠️ No records in {collection}. Skipping.")
            continue

        texts, meta_entries = [], []

        # ✅ Create one embedding per field-value pair
        for rec in data:
            if not isinstance(rec, dict):
                continue
            for field, value in rec.items():
                if value is None or str(value).strip() == "":
                    continue
                if isinstance(value, list):
                    values = value
                else:
                    values = [value]
                for v in values:
                    text = prepare_field_text(field, v)
                    texts.append(text)
                    meta_entries.append({"collection": collection, "field": field, "value": str(v)})

        if not texts:
            print(f"⚠️ No valid text found for {collection}.")
            continue

        print(f"🧩 Embedding {len(texts):,} field-value pairs...")
        embeddings = embed_texts(model, texts, BATCH_SIZE, DEVICE)

        # 🔹 Create new FAISS index for this collection
        dim = embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base_index)
        ids = np.arange(0, len(embeddings)).astype(np.int64)
        index.add_with_ids(embeddings, ids)
        faiss.write_index(index, faiss_path)
        print(f"✅ Saved FAISS index → {faiss_path}")

        # Update metadata and state
        for i, m in enumerate(meta_entries):
            m["id"] = int(ids[i])
            metadata.append(m)

        index_state[collection] = {
            "last_indexed": last_modified,
            "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "field_value_vectors": len(meta_entries)
        }

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(index_state, f, indent=2)

        print(f"✅ Indexed {collection}: {len(meta_entries):,} entries")

    print("\n🎯 All FAISS indexes built successfully.")
    print(f"📁 Stored in: {VECTOR_DIR}")
    print(f"🧾 Metadata: {META_PATH}")
    print(f"🕓 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# 🏁 MAIN GUARD (Windows-safe)
# ============================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    build_fieldwise_index()
