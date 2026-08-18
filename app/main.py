"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import ingest, query
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.services import build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the service container once at startup."""
    app.state.services = build_services(get_settings())
    yield


def create_app() -> FastAPI:
    """Assemble the FastAPI application with routers and exception handlers."""
    app = FastAPI(
        title="Multimodal RAG Service",
        description="REST API for PDF ingestion and retrieval-augmented Q&A "
        "with source citations.",
        version="0.1.0",
        lifespan=lifespan,
    )

    register_exception_handlers(app)
    app.include_router(ingest.router)
    app.include_router(query.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        services = app.state.services
        return {
            "status": "ok",
            "vector_db": services.settings.vector_db_type,
            "collection": services.settings.collection_name,
            "chunks_indexed": services.vector_store.count(),
            "documents": services.vector_store.list_documents(),
        }

    return app


app = create_app()
