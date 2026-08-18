"""Request/response schemas for the ingest endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class IngestResponse(BaseModel):
    """Result of indexing a single PDF document."""

    document: str
    pages: int
    element_counts: dict[str, int]
    chunks_indexed: int
    status: str
