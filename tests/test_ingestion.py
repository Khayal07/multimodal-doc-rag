"""Unit tests for the ingestion orchestrator (mocked parser/embeddings/store)."""

import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.loader import IngestionError, IngestionService
from app.ingestion.parser import ParsedDocument, ParsedElement


class _FakeParser:
    async def parse(self, file, document_name=None):
        return ParsedDocument(
            document=document_name or "doc.pdf",
            pages=2,
            elements=[ParsedElement(page=1, element_type="text", text="hello world")],
        )


class _FakeEmbeddings:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _FakeStore:
    def __init__(self):
        self.added = []
        self.deleted = []

    def delete_document(self, document):
        self.deleted.append(document)

    def add(self, chunks, embeddings):
        self.added.extend(chunks)


class _EmptyParser:
    async def parse(self, file, document_name=None):
        return ParsedDocument(document="empty.pdf", pages=0, elements=[])


class _ChunkerStub:
    def chunk(self, document):
        if not document.elements:
            return []
        return [
            Chunk(
                id="1",
                document=document.document,
                page=1,
                element_type="text",
                text="hello world",
                chunk_index=0,
            )
        ]


def _service(store, parser=None) -> IngestionService:
    return IngestionService(
        parser=parser or _FakeParser(),
        chunker=_ChunkerStub(),
        embeddings=_FakeEmbeddings(),
        vector_store=store,
    )


@pytest.mark.asyncio
async def test_ingest_indexes_chunks_and_reports_counts() -> None:
    store = _FakeStore()
    report = await _service(store).ingest(b"%PDF", document_name="upload.pdf")

    assert report.document == "upload.pdf"
    assert report.pages == 2
    assert report.chunks_indexed == 1
    assert report.element_counts == {"text": 1}
    assert store.deleted == ["upload.pdf"]
    assert len(store.added) == 1


@pytest.mark.asyncio
async def test_ingest_raises_when_no_content() -> None:
    with pytest.raises(IngestionError, match="No indexable content"):
        await _service(_FakeStore(), parser=_EmptyParser()).ingest(b"x", document_name="empty.pdf")


@pytest.mark.asyncio
async def test_parser_errors_propagate_to_caller() -> None:
    class _BoomParser:
        async def parse(self, file, document_name=None):
            from app.ingestion.parser import LlamaParseError

            raise LlamaParseError("LlamaParse returned 0 documents. Check API key.")

    with pytest.raises(Exception, match="0 documents"):
        await _service(_FakeStore(), parser=_BoomParser()).ingest(b"x", document_name="bad.pdf")
