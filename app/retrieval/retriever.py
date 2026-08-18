"""Retrieval: embed the query and pull the most relevant chunks."""

from __future__ import annotations

from app.embeddings.service import EmbeddingService
from app.vectorstore.base import BaseVectorStore, RetrievedChunk


class Retriever:
    """Converts a user query into a ranked list of source chunks."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: BaseVectorStore,
        default_top_k: int = 5,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document: str | None = None,
    ) -> list[RetrievedChunk]:
        """Embed ``query`` and return its ``top_k`` nearest chunks."""
        query_embedding = self._embeddings.embed_one(query)
        return self._vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k or self._default_top_k,
            document=document,
        )
