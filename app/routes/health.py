from fastapi import APIRouter, HTTPException

from app.dependencies import index, model

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    # Verify critical resources are loaded and available
    if model is None or index is None:
        raise HTTPException(status_code=503, detail="System not fully loaded")
    return {"status": "ready"}
