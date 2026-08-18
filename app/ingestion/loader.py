"""Ingestion orchestrator.

Wires the pipeline together: parse (LlamaParse) -> chunk -> embed -> store.
Re-ingesting a document replaces its previously indexed chunks. Parsing runs
async natively; the blocking embed/store steps are offloaded to a worker thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool

from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import Chunk, Chunker
from app.ingestion.parser import DocumentParser, FileInput
from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


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

    async def ingest(self, file: FileInput, document_name: str | None = None) -> IngestionReport:
        """Parse ``file`` and index every resulting chunk.

        Existing chunks for the same document are removed first so a re-upload
        always reflects the latest version of the file. Parser failures surface
        as :class:`app.ingestion.parser.LlamaParseError` (HTTP 500).
        """
        parsed = await self._parser.parse(file, document_name=document_name)

        chunks = self._chunker.chunk(parsed)
        if not chunks:
            raise IngestionError(f"No indexable content found in '{parsed.document}'.")

        await run_in_threadpool(self._index_chunks, parsed.document, chunks)

        logger.info(
            "Ingested '%s': %d pages, %d chunks (%s).",
            parsed.document,
            parsed.pages,
            len(chunks),
            ", ".join(f"{k}={v}" for k, v in parsed.count_by_type().items()),
        )
        return IngestionReport(
            document=parsed.document,
            pages=parsed.pages,
            element_counts=parsed.count_by_type(),
            chunks_indexed=len(chunks),
        )

    def _index_chunks(self, document: str, chunks: list[Chunk]) -> None:
        """Embed and persist chunks for ``document`` (blocking, worker thread)."""
        self._vector_store.delete_document(document)
        embeddings = self._embeddings.embed([chunk.text for chunk in chunks])
        self._vector_store.add(chunks, embeddings)
