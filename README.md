# Hariprasath-Infynd-Chatbot 🤖

A **Hybrid Retrieval-Augmented Generation (RAG)** chatbot API that combines FAISS vector search, BM25 lexical ranking, and fuzzy matching for intelligent document retrieval and LLM-powered summarization.

## Features ✨

- **Hybrid Search Scoring**: Combines embedding-based (E5), BM25 lexical, and fuzzy matching scores
- **Field-Aware Retrieval**: Automatically selects relevant fields based on query semantics
- **Automatic Query Expansion**: Uses concept embedding clustering to expand queries intelligently
- **Cross-Encoder Re-ranking** (Optional): High-accuracy re-ranking for improved relevance
- **Parallel Field Search**: Multi-threaded FAISS index querying for fast retrieval
- **LLM Summarization**: Ollama-based query intent summarization
- **REST API**: FastAPI endpoints for easy integration
- **Performance Metrics**: Detailed execution time breakdown per component

## Architecture 🏗️

```
Query Input
    ↓
[Query Expansion] (concept-based)
    ↓
[Field Selection] (embedding + keyword overlap)
    ↓
[FAISS Parallel Retrieval] (per-field vector search)
    ↓
[Hybrid Scoring] (embedding + BM25 + fuzzy)
    ↓
[Optional Cross-Encoder Re-ranking]
    ↓
[LLM Summarization] (Ollama)
    ↓
Structured Results + Recommendations
```

## Installation 📦

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional)
- [Ollama](https://ollama.ai) (for LLM summarization)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HariP-git/Hariprasath-Infynd-chatbot.git
   cd Hariprasath-Infynd-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (for LLM features)
   - Download from [ollama.ai](https://ollama.ai)
   - Pull the model: `ollama pull tinyllama`

### Data Setup

The system expects data in the following structure:

```
data/
├── vector/
│   ├── field1.faiss
│   ├── field2.faiss
│   └── ...
├── metadata.json
├── field_embeddings.json (auto-generated)
├── field_keywords.json (auto-generated)
├── concept_terms.json (auto-generated)
└── concept_embeddings.json (auto-generated)
```

**Building FAISS Indexes:**
```bash
python scripts/build_faiss_incremental.py
```

## Usage 🚀

### API Server

Start the FastAPI server:
```bash
python app.py --serve
```

Server runs on `http://localhost:8000`

### Query Endpoint

**POST** `/query`

Request:
```json
{
  "query": "What are the system requirements?",
  "mode": "fast"
}
```

Response:
```json
{
  "summary": "System needs at least 8GB RAM and Python 3.8+",
  "results": {
    "requirements": ["8GB RAM", "Python 3.8+"]
  },
  "recommendations": {
    "requirements": ["8GB RAM", "Python 3.8+", "CUDA 11.8+ (optional)"]
  },
  "model_used": "intfloat/e5-large-v2 + BM25",
  "execution_time_seconds": {
    "embedding": 0.025,
    "field_selection": 0.008,
    "retrieval": 0.015,
    "rerank": 0.0,
    "bm25_and_hybrid": 0.012,
    "llm": 0.420,
    "total": 0.480
  }
}
```

### Query Modes

- **`fast`** (default): Quick retrieval without query expansion
- **`high`**: Enhanced retrieval with query expansion and cross-encoder re-ranking (if available)

### CLI Testing

```bash
# Single query test
python app.py --query "What is the installation process?" --mode fast

# With high accuracy mode
python app.py --query "system requirements" --mode high
```

## Configuration ⚙️

Edit `app.py` to customize:

```python
# Model Selection
EMBED_MODEL = "intfloat/e5-large-v2"  # Embedding model
OLLAMA_MODEL = "tinyllama"             # LLM for summarization

# Search Parameters
TOP_K = 50                             # Results per field
HIGH_MATCH_SCORE = 0.90                # Strong match threshold
MEDIUM_MATCH_SCORE = 0.60              # Medium match threshold

# Hybrid Weights (must sum to 1.0)
WEIGHT_EMB = 0.6                       # Embedding score
WEIGHT_BM25 = 0.25                     # BM25 lexical score
WEIGHT_FUZZY = 0.15                    # Fuzzy match score

# Field Selection
FIELD_SIM_THRESHOLD = 0.35             # Min semantic similarity
MAX_RELEVANT_FIELDS = 12               # Max fields to search

# Performance
MAX_THREAD_POOL = 8                    # Parallel threads
RECOMMEND_LIMIT_PER_FIELD = 6          # Recommendations per field
```

## Key Components 🔧

### 1. **FAISS Indexes**
- Per-field vector indexes using `intfloat/e5-large-v2` embeddings
- Normalized L2 distances for cosine similarity

### 2. **BM25 Ranking**
- Lightweight implementation without external dependencies
- Term frequency-inverse document frequency scoring
- IDF calculation with Okapi BM25 formula

### 3. **Field Selection**
- Embedding similarity to field names
- Keyword overlap detection
- Automatic fallback to top fields

### 4. **Hybrid Scoring**
- Weighted combination of three scoring methods:
  - **Embedding**: Semantic similarity (60%)
  - **BM25**: Lexical matching (25%)
  - **Fuzzy**: String similarity via RapidFuzz (15%)

### 5. **Query Expansion**
- Concept term extraction from metadata
- Embedding-based nearest neighbor search
- Automatic query augmentation (high mode)

### 6. **LLM Summarization**
- Ollama integration for intent summarization
- Timeout protection (20 seconds default)
- Graceful fallback to heuristic summary

## Performance Metrics 📊

Typical execution times (on moderate hardware):

| Component | Time (ms) |
|-----------|-----------|
| Embedding | 25-50 |
| Field Selection | 5-15 |
| FAISS Retrieval | 10-30 |
| BM25 Scoring | 10-20 |
| LLM Summarization | 300-500 |
| **Total** | **~500-1000** |

## Requirements 📋

See `requirements.txt` for full dependencies:

- **FastAPI** 0.115.2 - Web framework
- **Uvicorn** 0.32.0 - ASGI server
- **Sentence-Transformers** 3.1.1 - Embeddings
- **PyTorch** ≥2.0.0 - Deep learning backend
- **FAISS** 1.8.0 - Vector search
- **NumPy** ≥1.26.0 - Numerical computing
- **RapidFuzz** 3.10.0 - Fuzzy matching
- **ijson** 3.2.3 - JSON streaming
- **Pydantic** ≥2.7.0 - Data validation

## Project Structure 📁

```
.
├── app.py                          # Main RAG API (FastAPI)
├── fallback.py                     # Alternative implementation
├── requirements.txt                # Python dependencies
├── scripts/
│   └── build_faiss_incremental.py # Index building script
├── data/
│   ├── vector/                     # FAISS index files
│   ├── metadata.json               # Field-value metadata
│   └── ...
└── README.md                       # This file
```

## Troubleshooting 🔧

### Issue: "No FAISS indexes found"
- Ensure `data/vector/` contains `.faiss` files
- Run `scripts/build_faiss_incremental.py` to generate indexes

### Issue: Ollama timeouts
- Check Ollama is running: `ollama serve`
- Increase `OLLAMA_TIMEOUT` in config (default: 20s)
- Summaries will fallback to heuristic if Ollama fails

### Issue: High latency
- Use `mode=fast` instead of `high`
- Reduce `TOP_K` for faster retrieval
- Ensure GPU is available (`torch.cuda.is_available()`)

### Issue: GPU Out of Memory
- Reduce batch size in `build_faiss_incremental.py`
- Use `faiss-cpu` instead of `faiss-gpu`
- Reduce embedding model (`e5-base-v2` instead of `e5-large-v2`)

## Advanced Usage 🚀

### Enabling Cross-Encoder Re-ranking

The system automatically detects and uses cross-encoders in high mode:

```bash
python app.py --query "complex query" --mode high
```

### Custom Field Weights

Boost specific fields by updating `FIELD_WEIGHTS`:

```python
FIELD_WEIGHTS = {
    "title": 1.2,
    "description": 1.0,
    "tags": 0.8
}
```

### Building Custom Indexes

```python
from scripts.build_faiss_incremental import build_fieldwise_index
build_fieldwise_index()
```

## Performance Optimization Tips ⚡

1. **Use GPU**: Ensure CUDA is installed and `torch.cuda.is_available()` returns `True`
2. **Batch Processing**: Query multiple inputs in parallel using ThreadPoolExecutor
3. **Caching**: Field embeddings and keywords are cached in JSON files
4. **Field Filtering**: Increase `FIELD_SIM_THRESHOLD` to search fewer fields
5. **Smaller Models**: Use `e5-base-v2` or `e5-small-v2` for faster inference





For issues, questions, or suggestions, please open an [issue](https://github.com/HariP-git/Hariprasath-Infynd-chatbot/issues) on GitHub.

---

**Made with ❤️ for intelligent information retrieval**
