"""Unit tests for configuration loading."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_required_credentials_must_be_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key=None, llama_cloud_api_key=None)  # type: ignore[arg-type]


def test_defaults_are_applied() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test",
        llama_cloud_api_key="llx-test",
    )
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.vector_db_type == "chroma"
    assert settings.chroma_persist_dir == "data/chroma"
    assert settings.collection_name == "multimodal_docs"
    assert settings.top_k == 5
    assert settings.llm_temperature == 0.0


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("TOP_K", "7")
    settings = Settings(_env_file=None, openai_api_key="sk-test", llama_cloud_api_key="llx-test")
    assert settings.llm_model == "gpt-4o"
    assert settings.top_k == 7


def test_invalid_vector_db_type_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_DB_TYPE", "pinecone")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="sk-test", llama_cloud_api_key="llx-test")


def test_persist_dir_is_created(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test",
        llama_cloud_api_key="llx-test",
        chroma_persist_dir=str(tmp_path / "nested" / "db"),
    )
    assert settings.persist_dir.is_dir()