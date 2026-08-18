"""Prompt construction and citation post-processing."""

from __future__ import annotations

import re

from app.ingestion.chunker import citation_label
from app.vectorstore.base import RetrievedChunk

SYSTEM_PROMPT = """You are an expert document assistant. Answer the user's question based ONLY on the provided context.

Rules:
- Base every statement exclusively on the context. Do not use outside knowledge.
- After every claim, cite the supporting source using its bracketed number, e.g. [1], [2].
- Multiple claims from different sources: append all applicable numbers, e.g. [1][3].
- If the context is insufficient to answer, say "I could not find enough information in the provided documents." and nothing else.
- Use markdown (headings, bullet lists, tables) when it improves readability. Preserve table data accurately.

Context:
{context}
"""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a numbered context block for the LLM.

    Numbering is stable: chunk ``n`` is referenced as ``[n]`` in citations and
    maps back to ``chunks[n - 1]`` during post-processing.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{index}] [{citation_label(chunk)}] (document: {chunk.document})\n{chunk.text}")
    return "\n\n".join(blocks)


_CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")
_CITATION_MARKER = re.compile(r"\[Source: .*?\]")


def resolve_citations(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Replace ``[n]`` markers in ``answer`` with full ``[Source: ...]`` labels.

    Only markers that map to a provided chunk are rewritten; any other bracketed
    number is left untouched.
    """

    def _replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 1 <= index <= len(chunks):
            return f"[Source: {citation_label(chunks[index - 1])}]"
        return match.group(0)

    return _CITATION_PATTERN.sub(_replace, answer)


def extract_citations(answer: str) -> list[str]:
    """Return every citation label found in ``answer`` in order of appearance."""
    return _CITATION_MARKER.findall(answer)