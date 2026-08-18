"""Unit tests for the ingestion orchestrator (mocked parser/embeddings/store)."""

from app.ingestion.chunker import Chunk
from app.ingestion.loader import IngestionService
from app.ingestion.parser import ParsedDocument, ParsedElement


class _FakeParser:
    def parse(self, file, document_name=None):
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


def test_ingest_indexes_chunks_and_reports_counts() -> None:
    store = _FakeStore()
    service = IngestionService(
        parser=_FakeParser(),
        chunker=_ChunkerStub(),
        embeddings=_FakeEmbeddings(),
        vector_store=store,
    )

    report = service.ingest(b"%PDF", document_name="upload.pdf")

    assert report.document == "upload.pdf"
    assert report.pages == 2
    assert report.chunks_indexed == 1
    assert report.element_counts == {"text": 1}
    assert store.deleted == ["upload.pdf"]
    assert len(store.added) == 1


def test_ingest_raises_when_no_content() -> None:
    class _EmptyParser:
        def parse(self, file, document_name=None):
            return ParsedDocument(document="empty.pdf", pages=0, elements=[])

    service = IngestionService(
        parser=_EmptyParser(),
        chunker=_ChunkerStub(),
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeStore(),
    )

    from app.ingestion.loader import IngestionError

    try:
        service.ingest(b"x", document_name="empty.pdf")
    except IngestionError as exc:
        assert "No indexable content" in str(exc)
    else:
        raise AssertionError("IngestionError was not raised")


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