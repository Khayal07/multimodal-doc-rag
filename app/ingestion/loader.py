"""Ingestion orchestrator.

Wires the pipeline together: parse (LlamaParse) -> chunk -> embed -> store.
Re-ingesting a document replaces its previously indexed chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import Chunker
from app.ingestion.parser import DocumentParser, FileInput
from app.vectorstore.base import BaseVectorStore


@dataclass(slots=True)
class IngestionReport:
    """Result summary returned for each ingested document."""

    document: str
    pages: int
    element_counts: dict[str, int]
    chunks_indexed: int
    status: str = "indexed"


class IngestionError(RuntimeError):
    """Raised when a document cannot be parsed or indexed."""


class IngestionService:
    """Coordinates the parse-to-store pipeline for a single PDF."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embeddings: EmbeddingService,
        vector_store: BaseVectorStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store

    def ingest(self, file: FileInput, document_name: str | None = None) -> IngestionReport:
        """Parse ``file`` and index every resulting chunk.

        Existing chunks for the same document are removed first so a re-upload
        always reflects the latest version of the file.
        """
        try:
            parsed = self._parser.parse(file, document_name=document_name)
        except Exception as exc:  # noqa: BLE001 - wrap and re-raise as domain error
            raise IngestionError(str(exc)) from exc

        chunks = self._chunker.chunk(parsed)
        if not chunks:
            raise IngestionError(f"No indexable content found in '{parsed.document}'.")

        self._vector_store.delete_document(parsed.document)
        embeddings = self._embeddings.embed([chunk.text for chunk in chunks])
        self._vector_store.add(chunks, embeddings)

        return IngestionReport(
            document=parsed.document,
            pages=parsed.pages,
            element_counts=parsed.count_by_type(),
            chunks_indexed=len(chunks),
        )