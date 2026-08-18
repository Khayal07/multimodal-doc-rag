"""LlamaParse integration.

Wraps the LlamaCloud parsing API (free tier) to convert complex PDFs into a
structured, element-aware representation. The JSON result mode returns every
element (heading, paragraph, table, list, image, ...) tagged with its page
number, which is what makes page-level citations possible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*llama-cloud.*")

from llama_parse import LlamaParse
from tenacity import retry, stop_after_attempt, wait_exponential

FileInput = str | Path | bytes

# Element types produced by LlamaParse JSON mode.
ELEMENT_TEXT = "text"
ELEMENT_HEADING = "heading"
ELEMENT_TABLE = "table"
ELEMENT_LIST = "list"
ELEMENT_IMAGE = "image"
ELEMENT_CODE = "code"


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


def _element_text(item: dict[str, Any]) -> str:
    """Best-effort extraction of the textual content of a parsed item."""
    if isinstance(item.get("md"), str) and item["md"].strip():
        return item["md"].strip()
    if isinstance(item.get("value"), str) and item["value"].strip():
        return item["value"].strip()
    if isinstance(item.get("text"), str) and item["text"].strip():
        return item["text"].strip()
    return ""


def _normalise_element_type(item: dict[str, Any]) -> str:
    """Map a raw LlamaParse item type to a stable element type."""
    raw_type = str(item.get("type", ELEMENT_TEXT)).lower()
    if raw_type in {"paragraph", "text"}:
        return ELEMENT_TEXT
    if raw_type in {"h1", "h2", "h3", "h4", "h5", "h6", "heading"}:
        return ELEMENT_HEADING
    return raw_type


class DocumentParser:
    """Thin, retry-safe wrapper around :class:`LlamaParse`."""

    def __init__(self, api_key: str, *, premium_mode: bool = False, num_workers: int = 4) -> None:
        self._parser = LlamaParse(
            api_key=api_key,
            result_type="json",
            premium_mode=premium_mode,
            num_workers=num_workers,
            verbose=False,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def parse(self, file: FileInput, document_name: str | None = None) -> ParsedDocument:
        """Parse a PDF and return a :class:`ParsedDocument`.

        ``document_name`` is used as the source label and defaults to the file
        name when ``file`` is a path.
        """
        if isinstance(file, (str, Path)):
            name = document_name or Path(file).name
        else:
            name = document_name or "uploaded.pdf"

        try:
            results = self._parser.get_json_result(file)
        except Exception as exc:
            raise LlamaParseError(f"LlamaParse failed for '{name}': {exc}") from exc

        pages = self._extract_pages(results, name)
        return ParsedDocument(
            document=name,
            pages=len(pages),
            elements=self._collect_elements(pages),
        )

    def _extract_pages(self, results: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        """Pull the ordered page list out of the raw JSON results."""
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("pages"), list):
                return result["pages"]
        raise LlamaParseError(f"LlamaParse returned no structured page data for '{name}'.")

    def _collect_elements(self, pages: list[dict[str, Any]]) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        for page in pages:
            page_number = int(page.get("page", 0))
            items = page.get("items") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                element_type = _normalise_element_type(item)
                if element_type == ELEMENT_IMAGE:
                    continue  # images have no searchable text of their own
                text = _element_text(item)
                if not text:
                    continue
                elements.append(
                    ParsedElement(
                        page=page_number,
                        element_type=element_type,
                        text=text,
                        raw=item,
                    )
                )
        return elements