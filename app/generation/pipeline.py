"""End-to-end question answering: retrieve, contextualize, generate, cite."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.generation.llm import LLMService
from app.generation.prompt import (
    SYSTEM_PROMPT,
    build_context,
    extract_citations,
    resolve_citations,
)
from app.retrieval.retriever import Retriever
from app.vectorstore.base import RetrievedChunk


@dataclass(slots=True)
class AnswerResult:
    """Raw pipeline output before serialization."""

    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


class GenerationPipeline:
    """Coordinates retrieval and generation for a single user query."""

    def __init__(self, retriever: Retriever, llm: LLMService) -> None:
        self._retriever = retriever
        self._llm = llm

    def answer(
        self,
        query: str,
        top_k: int,
        documents: list[str] | None = None,
    ) -> AnswerResult:
        """Retrieve evidence for ``query``, generate an answer, and resolve citations."""
        chunks = self._retrieve(query, top_k, documents)
        if not chunks:
            return AnswerResult(
                answer="I could not find enough information in the provided documents.",
                chunks=[],
                citations=[],
            )

        system_prompt = SYSTEM_PROMPT.format(context=build_context(chunks))
        raw_answer = self._llm.generate(system_prompt, query)
        cited_answer = resolve_citations(raw_answer, chunks)
        return AnswerResult(
            answer=cited_answer,
            chunks=chunks,
            citations=extract_citations(cited_answer),
        )

    def _retrieve(
        self,
        query: str,
        top_k: int,
        documents: list[str] | None,
    ) -> list[RetrievedChunk]:
        if not documents:
            return self._retriever.retrieve(query, top_k=top_k)

        merged: list[RetrievedChunk] = []
        for document in documents:
            merged.extend(self._retriever.retrieve(query, top_k=top_k, document=document))
        merged.sort(key=lambda chunk: chunk.score)
        return merged[:top_k]