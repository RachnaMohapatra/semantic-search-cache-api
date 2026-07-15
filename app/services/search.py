import re

from fastapi import HTTPException

import app.services.cache as cache_service
from app.dependencies import clean_documents, index, model


def search_products(query: str):
    # Remove unprintable characters (keep basic ASCII)
    query = re.sub(r"[^\x20-\x7E]", "", query)
    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400, detail="Query cannot be empty or just whitespace."
        )

    if query in cache_service.semantic_cache:
        cache_service.hit_count += 1
        cached_response = cache_service.semantic_cache[query].copy()
        cached_response["cache_hit"] = True
        return cached_response

    cache_service.miss_count += 1

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k=5)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append(
            {
                "rank": rank + 1,
                "document_id": int(idx),
                "text": clean_documents[idx][:300],
                "distance": float(distances[0][rank]),
            }
        )

    response = {
        "query": query,
        "cache_hit": False,
        "top_match_id": int(indices[0][0]),
        "distance": float(distances[0][0]),
        "results": results,
    }

    cache_service.semantic_cache[query] = response
    return response
