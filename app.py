from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import faiss

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
with open("clean_documents.txt", "r", encoding="utf-8") as f:
    clean_documents = [line.strip() for line in f.readlines()]

# Create embeddings
doc_embeddings = model.encode(clean_documents)

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Cache
semantic_cache = {}
hit_count = 0
miss_count = 0


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "FastAPI server running"}


@app.post("/query")
def query_endpoint(request: QueryRequest):

    global hit_count, miss_count

    query = request.query

    # Cache check
    if query in semantic_cache:
        hit_count += 1
        response = semantic_cache[query]
        response["cache_hit"] = True
        return response

    # Compute embedding
    query_embedding = model.encode([query])

    # FAISS search
    distances, indices = index.search(np.array(query_embedding), k=5)

    results = [
        {
            "text": clean_documents[i][:250],
            "distance": float(distances[0][idx])
        }
        for idx, i in enumerate(indices[0])
    ]

    response = {
        "query": query,
        "cache_hit": False,
        "similarity_score": float(1 - distances[0][0]),
        "dominant_cluster": 0,
        "results": results
    }

    semantic_cache[query] = response
    miss_count += 1

    return response


@app.get("/cache/stats")
def cache_stats():

    total = hit_count + miss_count
    hit_rate = (hit_count / total) * 100 if total > 0 else 0

    return {
        "total_entries": len(semantic_cache),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": f"{hit_rate:.2f}%"
    }


@app.delete("/cache")
def clear_cache():

    global semantic_cache, hit_count, miss_count

    semantic_cache = {}
    hit_count = 0
    miss_count = 0

    return {"message": "Cache cleared"}