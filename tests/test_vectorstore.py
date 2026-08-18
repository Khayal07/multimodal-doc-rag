"""Integration tests for the ChromaDB-backed vector store."""

from app.ingestion.chunker import Chunk
from app.vectorstore.chroma_store import ChromaStore


def _store(tmp_path) -> ChromaStore:
    return ChromaStore(persist_dir=str(tmp_path), collection_name="test_collection")


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="a-p0", document="doc.pdf", page=1, element_type="text",
            text="alpha beta gamma", chunk_index=0,
        ),
        Chunk(
            id="b-p1", document="doc.pdf", page=3, element_type="table",
            text="|x|y|", chunk_index=1, table_id=2,
        ),
    ]


def test_add_query_count_roundtrip(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert store.count() == 2
    assert store.list_documents() == ["doc.pdf"]

    results = store.query([1.0, 0.0, 0.0], top_k=2)
    assert results[0].id == "a-p0"
    assert results[0].page == 1
    assert results[0].table_id is None
    assert results[0].text == "alpha beta gamma"


def test_query_returns_table_metadata(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    results = store.query([0.0, 1.0, 0.0], top_k=2)
    assert results[0].id == "b-p1"
    assert results[0].element_type == "table"
    assert results[0].table_id == 2


def _other_chunk() -> list[Chunk]:
    return [
        Chunk(
            id="c-p0", document="other.pdf", page=1, element_type="text",
            text="unrelated", chunk_index=0,
        )
    ]


def test_query_filtered_by_document(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.add(_other_chunk(), [[1.0, 0.0, 0.0]])

    results = store.query([1.0, 0.0, 0.0], top_k=5, document="other.pdf")
    assert len(results) == 1
    assert results[0].document == "other.pdf"


def test_delete_document_removes_only_its_chunks(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.add(_other_chunk(), [[0.5, 0.5, 0.0]])

    store.delete_document("doc.pdf")
    assert store.count() == 1
    assert store.list_documents() == ["other.pdf"]


def test_upsert_replaces_existing_chunks(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_chunks(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    updated = Chunk(
        id="a-p0", document="doc.pdf", page=1, element_type="text",
        text="new version", chunk_index=0,
    )
    store.add([updated], [[1.0, 0.0, 0.0]])

    results = store.query([1.0, 0.0, 0.0], top_k=5)
    assert store.count() == 2
    assert results[0].text == "new version"
