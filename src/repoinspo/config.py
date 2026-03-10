"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    github_token: SecretStr | None = Field(default=None, alias="GITHUB_TOKEN")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    llm_models: list[str] = Field(
        default_factory=lambda: ["anthropic/claude-sonnet-4-20250514"],
        alias="LLM_MODELS",
    )
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    cache_dir: Path = Field(default=Path(".cache") / "repoinspo", alias="CACHE_DIR")
    max_tokens: int = Field(default=8192, alias="MAX_TOKENS", ge=1)
    default_token_budget: int = Field(default=100000, alias="DEFAULT_TOKEN_BUDGET", ge=1)
    council_enabled: bool = Field(default=True, alias="COUNCIL_ENABLED")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @field_validator("llm_models", mode="before")
    @classmethod
    def _parse_llm_models(cls, value: Any) -> list[str]:
        if value is None:
            return ["anthropic/claude-sonnet-4-20250514"]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("LLM_MODELS must be a comma-separated string or list of strings")

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _expand_cache_dir(cls, value: Any) -> Path:
        return Path(value).expanduser() if value is not None else Path(".cache") / "repoinspo"

    @property
    def api_keys(self) -> dict[str, str]:
        keys: dict[str, str] = {}
        if self.anthropic_api_key:
            keys["anthropic"] = self.anthropic_api_key.get_secret_value()
        if self.openai_api_key:
            keys["openai"] = self.openai_api_key.get_secret_value()
        if self.google_api_key:
            keys["google"] = self.google_api_key.get_secret_value()
        return keys

    @property
    def council_mode(self) -> bool:
        return self.council_enabled and len(self.llm_models) > 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
