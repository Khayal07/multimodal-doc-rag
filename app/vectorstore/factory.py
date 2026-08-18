"""Vector store factory resolving the configured backend."""

from __future__ import annotations

from app.config import Settings
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.chroma_store import ChromaStore


def build_vector_store(settings: Settings) -> BaseVectorStore:
    """Return the vector store implementation selected by ``VECTOR_DB_TYPE``."""
    if settings.vector_db_type == "chroma":
        return ChromaStore(
            persist_dir=str(settings.persist_dir),
            collection_name=settings.collection_name,
        )
    if settings.vector_db_type == "qdrant":
        raise NotImplementedError("Qdrant backend is not implemented yet.")
    raise ValueError(f"Unsupported VECTOR_DB_TYPE: {settings.vector_db_type}")