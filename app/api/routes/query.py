"""Query API: answer a question using the indexed documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_services
from app.schemas.query import QueryRequest, QueryResponse, SourceRef
from app.services import Services

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def answer_query(
    payload: QueryRequest,
    services: Services = Depends(get_services),  # noqa: B008
) -> QueryResponse:
    """Retrieve relevant chunks, generate a cited answer, and return the evidence."""
    top_k = payload.top_k or services.settings.top_k
    result = await run_in_threadpool(
        services.pipeline.answer,
        payload.query,
        top_k,
        payload.documents,
    )
    return QueryResponse(
        query=payload.query,
        answer=result.answer,
        sources=[
            SourceRef(
                chunk_id=chunk.id,
                document=chunk.document,
                page=chunk.page,
                element_type=chunk.element_type,
                table_id=chunk.table_id,
                text=chunk.text,
                score=chunk.score,
            )
            for chunk in result.chunks
        ],
        citations=result.citations,
    )