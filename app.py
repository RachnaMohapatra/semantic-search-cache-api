from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ==================================================
# FastAPI App
# ==================================================

app = FastAPI(
    title="Semantic Product Search Engine",
    description="Semantic Product Search using Sentence Transformers, FAISS and Intelligent Caching",
    version="1.0.0"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# Load Resources
# ==================================================

print("Loading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading processed documents...")

with open("clean_documents.txt", "r", encoding="utf-8") as f:
    clean_documents = [line.strip() for line in f]

print(f"Loaded {len(clean_documents)} documents")

print("Loading embeddings...")
doc_embeddings = np.load("product_embeddings.npy")

print("Loading FAISS index...")
index = faiss.read_index("product_index.faiss")

print("System Ready!")

# ==================================================
# Semantic Cache
# ==================================================

semantic_cache = {}

hit_count = 0
miss_count = 0

# ==================================================
# Request Schema
# ==================================================

class QueryRequest(BaseModel):
    query: str

# ==================================================
# Home Endpoint
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Semantic Product Search API Running",
        "documents": len(clean_documents),
        "cache_entries": len(semantic_cache)
    }

# ==================================================
# Search Endpoint
# ==================================================

@app.post("/query")
def query_endpoint(request: QueryRequest):

    global hit_count
    global miss_count

    query = request.query.strip()

    # ---------------------------------------------
    # CACHE HIT
    # ---------------------------------------------

    if query in semantic_cache:

        hit_count += 1

        cached_response = semantic_cache[query].copy()
        cached_response["cache_hit"] = True

        return cached_response

    # ---------------------------------------------
    # CACHE MISS
    # ---------------------------------------------

    miss_count += 1

    query_embedding = model.encode(
        [query]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k=5
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        results.append({
            "rank": rank + 1,
            "document_id": int(idx),
            "text": clean_documents[idx][:300],
            "distance": float(distances[0][rank])
        })

    response = {
        "query": query,
        "cache_hit": False,
        "top_match_id": int(indices[0][0]),
        "distance": float(distances[0][0]),
        "results": results
    }

    semantic_cache[query] = response

    return response

# ==================================================
# Cache Statistics
# ==================================================

@app.get("/cache/stats")
def cache_stats():

    total_queries = hit_count + miss_count

    hit_rate = (
        (hit_count / total_queries) * 100
        if total_queries > 0
        else 0
    )

    return {
        "total_queries": total_queries,
        "cache_entries": len(semantic_cache),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": f"{hit_rate:.2f}%"
    }

# ==================================================
# Clear Cache
# ==================================================

@app.delete("/cache")
def clear_cache():

    global semantic_cache
    global hit_count
    global miss_count

    semantic_cache.clear()

    hit_count = 0
    miss_count = 0

    return {
        "message": "Cache cleared successfully"
    }