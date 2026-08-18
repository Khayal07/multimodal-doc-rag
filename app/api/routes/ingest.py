"""Ingestion API: upload a PDF and index it."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_services
from app.schemas.ingest import IngestResponse
from app.services import Services

router = APIRouter(prefix="/api/v1", tags=["ingest"])

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MiB


@router.post("/ingest/pdf", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to parse and index."),  # noqa: B008
    services: Services = Depends(get_services),  # noqa: B008
) -> IngestResponse:
    """Parse a PDF with LlamaParse and index every chunk into the vector store."""
    if file.content_type and not file.content_type.startswith("application/pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only application/pdf files are supported.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the 25 MiB limit.",
        )

    report = await services.ingestion.ingest(content, file.filename)
    return IngestResponse(**asdict(report))
