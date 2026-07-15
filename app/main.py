from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.services.cache as cache_service
from app.dependencies import clean_documents
from app.routes import cache, health, search

app = FastAPI(
    title="Semantic Product Search Engine",
    description=(
        "Semantic Product Search using Sentence Transformers, "
        "FAISS and Intelligent Caching"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(cache.router)
app.include_router(health.router)


@app.get("/")
def home():
    return {
        "message": "Semantic Product Search API Running",
        "documents": len(clean_documents),
        "cache_entries": len(cache_service.semantic_cache),
    }
