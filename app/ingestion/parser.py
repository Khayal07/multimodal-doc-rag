"""LlamaParse integration.

Wraps the LlamaCloud parsing API (free tier) to convert complex PDFs into a
page-aware, element-structured representation. Markdown result mode is used for
the most reliable extraction; pages are split server-side and elements (tables,
headings, paragraphs) are then re-detected from each page's markdown so that
page- and table-level citations remain possible.

Parsing is fully async so it never blocks or conflicts with FastAPI's event
loop (``nest_asyncio`` is applied defensively at import time).
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nest_asyncio

nest_asyncio.apply()

warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*llama-cloud.*")

from llama_parse import LlamaParse  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_exponential  # noqa: E402

logger = logging.getLogger(__name__)

FileInput = str | Path | bytes

# Element types produced by the markdown block classifier.
ELEMENT_TEXT = "text"
ELEMENT_HEADING = "heading"
ELEMENT_TABLE = "table"
ELEMENT_LIST = "list"
ELEMENT_IMAGE = "image"
ELEMENT_CODE = "code"

_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S")


@dataclass(slots=True)
class ParsedElement:
    """A single page-bound element extracted from a PDF."""

    page: int
    element_type: str
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    """Structured result of parsing one PDF."""

    document: str
    pages: int
    elements: list[ParsedElement] = field(default_factory=list)

    def count_by_type(self) -> dict[str, int]:
        """Return a ``{element_type: count}`` summary across all pages."""
        counts: dict[str, int] = {}
        for element in self.elements:
            counts[element.element_type] = counts.get(element.element_type, 0) + 1
        return counts


class LlamaParseError(RuntimeError):
    """Raised when LlamaParse cannot produce a parseable result."""


def _classify_block(block: str) -> tuple[str, str]:
    """Classify a markdown block into ``(element_type, clean_text)``."""
    non_empty = [line for line in block.split("\n") if line.strip()]
    is_table = (
        len(non_empty) >= 2
        and all("|" in line for line in non_empty)
        and any(_TABLE_SEPARATOR.match(line) and "-" in line for line in non_empty)
    )
    if is_table:
        return ELEMENT_TABLE, block.strip()
    stripped = block.strip()
    if not stripped:
        return "", ""
    if _HEADING_PATTERN.match(stripped):
        return ELEMENT_HEADING, stripped
    if stripped.startswith("- ") or stripped.startswith("* "):
        return ELEMENT_LIST, stripped
    return ELEMENT_TEXT, stripped


def _page_elements(markdown: str, page_number: int) -> list[ParsedElement]:
    """Split one page's markdown into classified :class:`ParsedElement`\\ s."""
    elements: list[ParsedElement] = []
    for block in re.split(r"\n\s*\n", markdown):
        element_type, text = _classify_block(block)
        if element_type and text:
            elements.append(ParsedElement(page=page_number, element_type=element_type, text=text))
    return elements


class DocumentParser:
    """Async, retry-safe wrapper around :class:`LlamaParse` (markdown mode)."""

    def __init__(self, api_key: str, *, premium_mode: bool = False, num_workers: int = 4) -> None:
        self._parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            premium_mode=premium_mode,
            num_workers=num_workers,
            verbose=True,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def parse(self, file: FileInput, document_name: str | None = None) -> ParsedDocument:
        """Parse a PDF (via ``await parser.aload_data``) into a :class:`ParsedDocument`.

        ``document_name`` is used as the source label and defaults to the file
        name when ``file`` is a path.
        """
        if isinstance(file, (str, Path)):
            name = document_name or Path(file).name
        else:
            name = document_name or "uploaded.pdf"

        logger.info("LlamaParse: parsing '%s' (markdown mode)...", name)
        try:
            documents = await self._parser.aload_data(file, extra_info={"file_name": name})
        except Exception as exc:
            logger.error("LlamaParse: parsing '%s' failed: %s", name, exc)
            raise LlamaParseError(f"LlamaParse failed for '{name}': {exc}") from exc

        logger.info("LlamaParse: '%s' returned %d page document(s).", name, len(documents))
        if not documents:
            raise LlamaParseError(
                "LlamaParse returned 0 documents. Check API key and LlamaCloud dashboard limits."
            )

        elements: list[ParsedElement] = []
        for page_number, page in enumerate(documents, start=1):
            markdown = getattr(page, "text", "") or ""
            elements.extend(_page_elements(markdown, page_number))

        if not elements:
            raise LlamaParseError(
                f"LlamaParse parsed '{name}' but no extractable content was found."
            )

        logger.info(
            "LlamaParse: '%s' -> %d pages, %d elements (%s).",
            name,
            len(documents),
            len(elements),
            ", ".join(f"{k}={v}" for k, v in _count_by_type(elements).items()),
        )
        return ParsedDocument(document=name, pages=len(documents), elements=elements)


def _count_by_type(elements: list[ParsedElement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.element_type] = counts.get(element.element_type, 0) + 1
    return counts
