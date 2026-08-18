"""Unit tests for the document chunker."""

from app.ingestion.chunker import Chunker, citation_label
from app.ingestion.parser import ParsedDocument, ParsedElement


def _document(elements: list[ParsedElement]) -> ParsedDocument:
    return ParsedDocument(document="report.pdf", pages=3, elements=elements)


def test_text_element_is_split_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunker = Chunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(_document([ParsedElement(page=1, element_type="text", text=text)]))

    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.element_type == "text" for c in chunks)
    assert all(c.page == 1 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_short_text_stays_single_chunk() -> None:
    chunker = Chunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(_document([ParsedElement(page=1, element_type="text", text="Short.")]))
    assert len(chunks) == 1
    assert chunks[0].text == "Short."


def test_tables_kept_whole_and_numbered_per_page() -> None:
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    chunker = Chunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk(
        _document(
            [
                ParsedElement(page=2, element_type="table", text=table),
                ParsedElement(page=3, element_type="table", text=table),
                ParsedElement(page=3, element_type="table", text=table),
            ]
        )
    )

    assert len(chunks) == 3
    assert all(c.text == table for c in chunks)
    assert [c.table_id for c in chunks] == [1, 1, 2]
    assert citation_label(chunks[1]) == "Page 3, Table 1"
    assert citation_label(chunks[2]) == "Page 3, Table 2"


def test_headings_are_kept_whole() -> None:
    heading = "Chapter 5: Results"
    chunker = Chunker(chunk_size=5, chunk_overlap=0)
    chunks = chunker.chunk(_document([ParsedElement(page=4, element_type="heading", text=heading)]))
    assert len(chunks) == 1
    assert chunks[0].text == heading


def test_empty_text_is_skipped() -> None:
    chunker = Chunker()
    chunks = chunker.chunk(
        _document([ParsedElement(page=1, element_type="text", text="   "), ParsedElement(page=1, element_type="table", text="|a|")])
    )
    assert len(chunks) == 1
    assert chunks[0].element_type == "table"