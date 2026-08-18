"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingestion.chunker import Chunk
from app.vectorstore.base import BaseVectorStore, RetrievedChunk


class ChromaStore(BaseVectorStore):
    """Persistent ChromaDB store with cosine-distance similarity search."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._to_metadata(chunk) for chunk in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        document: str | None = None,
    ) -> list[RetrievedChunk]:
        where = {"document": {"$eq": document}} if document else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        return self._from_query_result(result)

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[str]:
        documents = self._collection.get(include=["metadatas"]).get("metadatas") or []
        return sorted({meta["document"] for meta in documents if isinstance(meta, dict)})

    def delete_document(self, document: str) -> None:
        result = self._collection.get(where={"document": {"$eq": document}}, include=[])
        ids = result.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)

    @staticmethod
    def _to_metadata(chunk: Chunk) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "document": chunk.document,
            "page": chunk.page,
            "element_type": chunk.element_type,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.table_id is not None:
            metadata["table_id"] = chunk.table_id
        return metadata

    @staticmethod
    def _from_query_result(result: dict[str, Any]) -> list[RetrievedChunk]:
        ids = result.get("ids") or [[]]
        metadatas = result.get("metadatas") or [[]]
        documents = result.get("documents") or [[]]
        distances = result.get("distances") or [[]]

        chunks: list[RetrievedChunk] = []
        for row_index in range(len(ids[0])):
            meta = metadatas[0][row_index] if metadatas and metadatas[0] else {}
            chunks.append(
                RetrievedChunk(
                    id=ids[0][row_index],
                    document=meta.get("document", ""),
                    page=int(meta.get("page", 0)),
                    element_type=meta.get("element_type", "text"),
                    text=documents[0][row_index] if documents and documents[0] else "",
                    chunk_index=int(meta.get("chunk_index", 0)),
                    table_id=int(meta["table_id"]) if meta.get("table_id") is not None else None,
                    score=float(distances[0][row_index]) if distances and distances[0] else 0.0,
                )
            )
        return chunks
