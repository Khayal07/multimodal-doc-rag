"""Unit tests for citation formatting and post-processing."""

from app.generation.prompt import build_context, extract_citations, resolve_citations
from app.vectorstore.base import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id="a", document="r.pdf", page=3, element_type="table",
            text="data", chunk_index=0, table_id=1,
        ),
        RetrievedChunk(
            id="b", document="r.pdf", page=5, element_type="text",
            text="more", chunk_index=1,
        ),
    ]


def test_build_context_numbers_chunks() -> None:
    context = build_context(_chunks())
    assert "[1] [Page 3, Table 1]" in context
    assert "[2] [Page 5]" in context
    assert "document: r.pdf" in context


def test_resolve_citations_replaces_known_markers() -> None:
    answer = "From the data [1] and the text [2], we conclude."
    resolved = resolve_citations(answer, _chunks())
    assert resolved == (
        "From the data [Source: Page 3, Table 1] and the text "
        "[Source: Page 5], we conclude."
    )


def test_resolve_citations_leaves_unknown_markers() -> None:
    resolved = resolve_citations("See [7] for details [1]", _chunks())
    assert resolved == "See [7] for details [Source: Page 3, Table 1]"


def test_extract_citations_returns_labels_in_order() -> None:
    answer = "First [Source: Page 3, Table 1], then [Source: Page 5]."
    assert extract_citations(answer) == ["[Source: Page 3, Table 1]", "[Source: Page 5]"]


def test_resolve_citations_handles_missing_answer_markers() -> None:
    assert resolve_citations("No citations here.", _chunks()) == "No citations here."
