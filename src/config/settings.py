"""Configuration centralisée via Pydantic Settings."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class EmbeddingProvider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"


class JiraAuthType(str, Enum):
    TOKEN = "token"
    OAUTH2 = "oauth2"


class JiraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIRA_", extra="ignore")

    base_url: str = Field(..., description="URL de base de l'instance Jira")
    email: str = Field(..., description="Email du compte de service")
    api_token: SecretStr = Field(..., description="API Token ou PAT Jira")
    auth_type: JiraAuthType = JiraAuthType.TOKEN
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[SecretStr] = None
    oauth_token_url: str = "https://auth.atlassian.com/oauth/token"
    request_timeout: int = 30
    max_retries: int = 4

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class RovoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROVO_", extra="ignore")

    api_base_url: str = "https://api.atlassian.com/rovo/v1"
    api_token: Optional[SecretStr] = None
    enabled: bool = True


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_api_key: Optional[SecretStr] = None
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 8192
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[SecretStr] = None
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-01"


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    model: str = "intfloat/multilingual-e5-large"
    dimension: int = 1024


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", extra="ignore")

    host: str = "localhost"
    port: int = 6333
    api_key: Optional[SecretStr] = None
    collection_architecture: str = "architecture_docs"
    collection_governance: str = "governance_docs"
    collection_jira: str = "jira_tickets"
    collection_deliverables: str = "deliverables"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    db: str = "techoffice_cockpit"
    user: str = "techoffice"
    password: SecretStr = SecretStr("")
    database_url: Optional[str] = Field(None, alias="DATABASE_URL")

    @property
    def async_url(self) -> str:
        if self.database_url:
            return self.database_url
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"


class MinIOSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINIO_", extra="ignore")

    endpoint: str = "localhost:9000"
    access_key: SecretStr = SecretStr("")
    secret_key: SecretStr = SecretStr("")
    bucket_deliverables: str = "deliverables"
    bucket_documents: str = "documents"
    secure: bool = False


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    score_threshold: float = 0.65


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    app_secret_key: SecretStr = SecretStr("dev-secret-change-in-production")
    app_log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[SecretStr] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = False

    @property
    def jira(self) -> JiraSettings:
        return JiraSettings()

    @property
    def rovo(self) -> RovoSettings:
        return RovoSettings()

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings()

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings()

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings()

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings()

    @property
    def minio(self) -> MinIOSettings:
        return MinIOSettings()

    @property
    def rag(self) -> RAGSettings:
        return RAGSettings()

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnvironment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        try:
            s = get_settings()
            print(f"Configuration valide. Environnement : {s.app_env}")
            sys.exit(0)
        except Exception as e:
            print(f"Erreur de configuration : {e}", file=sys.stderr)
            sys.exit(1)
