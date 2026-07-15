from fastapi import APIRouter

import app.services.search as search_service
from app.models.request import QueryRequest

router = APIRouter()


@router.post("/query")
def query_endpoint(request: QueryRequest):
    return search_service.search_products(request.query)
