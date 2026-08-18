"""OpenAI embedding client used to vectorize document chunks."""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

_BATCH_SIZE = 100


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class EmbeddingService:
    """Thin, retry-safe wrapper around the OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str, max_retries: int = 3) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        reraise=True,
    )
    def _create(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:  # noqa: BLE001 - surface any provider failure
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
        return [item.embedding for item in response.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving input order."""
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            results.extend(self._create(batch))
        return results

    def embed_one(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._create([text])[0]