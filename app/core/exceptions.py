"""Domain exceptions and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.embeddings.service import EmbeddingError
from app.generation.llm import LLMError
from app.ingestion.loader import IngestionError
from app.ingestion.parser import LlamaParseError


class NotFoundError(RuntimeError):
    """Raised when a requested resource does not exist."""


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON handlers for every domain exception."""

    @app.exception_handler(LlamaParseError)
    async def _llama_parse_handler(_: Request, exc: LlamaParseError) -> JSONResponse:
        return _error_response(500, str(exc))

    @app.exception_handler(IngestionError)
    async def _ingestion_handler(_: Request, exc: IngestionError) -> JSONResponse:
        return _error_response(400, str(exc))

    @app.exception_handler(EmbeddingError)
    async def _embedding_handler(_: Request, exc: EmbeddingError) -> JSONResponse:
        return _error_response(502, str(exc))

    @app.exception_handler(LLMError)
    async def _llm_handler(_: Request, exc: LLMError) -> JSONResponse:
        return _error_response(502, str(exc))

    @app.exception_handler(NotFoundError)
    async def _not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return _error_response(404, str(exc))
