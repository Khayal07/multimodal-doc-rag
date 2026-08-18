"""Service container: builds every pipeline component from settings."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.embeddings.service import EmbeddingService
from app.generation.llm import LLMService
from app.generation.pipeline import GenerationPipeline
from app.ingestion.chunker import Chunker
from app.ingestion.loader import IngestionService
from app.ingestion.parser import DocumentParser
from app.retrieval.retriever import Retriever
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.factory import build_vector_store


@dataclass(slots=True)
class Services:
    """All runtime components, wired together and ready for injection."""

    settings: Settings
    parser: DocumentParser
    embeddings: EmbeddingService
    vector_store: BaseVectorStore
    chunker: Chunker
    ingestion: IngestionService
    retriever: Retriever
    llm: LLMService
    pipeline: GenerationPipeline


def build_services(settings: Settings) -> Services:
    """Construct the full service graph from ``settings``."""
    parser = DocumentParser(api_key=settings.llama_cloud_api_key.get_secret_value())
    embeddings = EmbeddingService(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.embedding_model,
        max_retries=settings.max_retries,
    )
    vector_store = build_vector_store(settings)
    chunker = Chunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    ingestion = IngestionService(parser, chunker, embeddings, vector_store)
    retriever = Retriever(embeddings, vector_store, default_top_k=settings.top_k)
    llm = LLMService(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    pipeline = GenerationPipeline(retriever, llm)
    return Services(
        settings=settings,
        parser=parser,
        embeddings=embeddings,
        vector_store=vector_store,
        chunker=chunker,
        ingestion=ingestion,
        retriever=retriever,
        llm=llm,
        pipeline=pipeline,
    )
