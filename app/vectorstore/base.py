"""Abstract vector store contract.

Concrete backends (Chroma, and later Qdrant) implement this interface so the
rest of the application stays decoupled from any single vector database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ingestion.chunker import Chunk


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by a similarity search, enriched with its score."""

    id: str
    document: str
    page: int
    element_type: str
    text: str
    chunk_index: int
    table_id: int | None = field(default=None)
    score: float = field(default=0.0)


class BaseVectorStore(ABC):
    """Interface every vector store implementation must satisfy."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist ``chunks`` together with their precomputed ``embeddings``."""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        document: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, optionally filtered by document."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored chunks."""

    @abstractmethod
    def list_documents(self) -> list[str]:
        """Return the distinct document names present in the store."""

    @abstractmethod
    def delete_document(self, document: str) -> None:
        """Remove every chunk belonging to ``document``."""
