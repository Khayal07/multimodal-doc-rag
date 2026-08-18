"""Request/response schemas for the query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A user question plus optional retrieval tuning."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50, description="Chunks to retrieve (overrides default).")
    documents: list[str] | None = Field(default=None, description="Restrict search to these documents.")


class SourceRef(BaseModel):
    """A retrieved chunk surfaced to the caller."""

    chunk_id: str
    document: str
    page: int
    element_type: str
    table_id: int | None = None
    text: str
    score: float


class QueryResponse(BaseModel):
    """The generated answer together with its evidence."""

    query: str
    answer: str
    sources: list[SourceRef]
    citations: list[str]