from typing import Any

semantic_cache: dict[str, Any] = {}
hit_count: int = 0
miss_count: int = 0


def get_cache_stats():
    global hit_count, miss_count, semantic_cache
    total_queries = hit_count + miss_count
    hit_rate = ((hit_count / total_queries) * 100) if total_queries > 0 else 0
    return {
        "total_queries": total_queries,
        "cache_entries": len(semantic_cache),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": f"{hit_rate:.2f}%",
    }


def clear_cache():
    global semantic_cache, hit_count, miss_count
    semantic_cache.clear()
    hit_count = 0
    miss_count = 0
    return {"message": "Cache cleared successfully"}
