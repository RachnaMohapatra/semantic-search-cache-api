from fastapi import APIRouter

import app.services.cache as cache_service

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats")
def cache_stats():
    return cache_service.get_cache_stats()


@router.delete("")
def clear_cache():
    return cache_service.clear_cache()
