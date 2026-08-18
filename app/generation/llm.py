"""LLM wrapper for answer generation via OpenAI chat completions."""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMError(RuntimeError):
    """Raised when answer generation fails."""


class LLMService:
    """Thin, retry-safe wrapper around the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0) -> None:
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's answer for the given system and user prompts."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
            )
        except Exception as exc:
            raise LLMError(f"Chat completion failed: {exc}") from exc
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("Chat completion returned an empty response.")
        return content.strip()
