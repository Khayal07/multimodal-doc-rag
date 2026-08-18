"""Document chunking.

Converts parsed elements into retrieval-ready chunks. Text-bearing elements are
split on natural boundaries (paragraphs/headings) with configurable overlap.
Tables are never split: they are stored whole and receive a per-page ``table_id``
so citations can point at e.g. "Page 3, Table 1".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.ingestion.parser import (
    ELEMENT_HEADING,
    ELEMENT_TABLE,
    ParsedDocument,
)

_CHUNK_ID_PREFIX_MAX = 16


@dataclass(slots=True)
class Chunk:
    """A self-contained, indexable unit of document content."""

    id: str
    document: str
    page: int
    element_type: str
    text: str
    chunk_index: int
    table_id: int | None = field(default=None)


def _document_prefix(document: str) -> str:
    """Stable short prefix for chunk ids derived from the file name."""
    digest = hashlib.sha1(document.encode("utf-8")).hexdigest()[:_CHUNK_ID_PREFIX_MAX]
    return digest


def _split_on_boundaries(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping segments no longer than ``size``.

    Boundaries are preferred at paragraph breaks (``\\n\\n``) or single newlines
    and otherwise fall back to word boundaries.
    """
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + size, len(text))
        if end < len(text):
            window = text[cursor:end]
            boundary = max(
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind(" "),
            )
            if boundary > 0:
                end = cursor + boundary + (1 if window[boundary] in {"\n", " "} else 0)
        piece = text[cursor:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        cursor = end - overlap
    return pieces


class Chunker:
    """Produces citation-aware chunks from a :class:`ParsedDocument`."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Convert all elements of ``document`` into :class:`Chunk` instances."""
        chunks: list[Chunk] = []
        prefix = _document_prefix(document.document)
        table_ids: dict[int, int] = {}
        global_index = 0

        for element in document.elements:
            table_id = None
            if element.element_type == ELEMENT_TABLE:
                table_ids[element.page] = table_ids.get(element.page, 0) + 1
                table_id = table_ids[element.page]
                pieces = [element.text]
            elif element.element_type == ELEMENT_HEADING:
                pieces = [element.text]
            else:
                pieces = _split_on_boundaries(element.text, self.chunk_size, self.chunk_overlap)

            for piece in pieces:
                chunks.append(
                    Chunk(
                        id=f"{prefix}-p{element.page}-{global_index}",
                        document=document.document,
                        page=element.page,
                        element_type=element.element_type,
                        text=piece,
                        chunk_index=global_index,
                        table_id=table_id,
                    )
                )
                global_index += 1

        return chunks


def citation_label(chunk: Chunk) -> str:
    """Human-readable citation label, e.g. ``"Page 3, Table 1"``."""
    if chunk.table_id is not None:
        return f"Page {chunk.page}, Table {chunk.table_id}"
    return f"Page {chunk.page}"