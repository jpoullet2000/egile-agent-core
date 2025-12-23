"""Configuration management for Egile Agent Core."""

from pydantic_settings import BaseSettings


class EgileConfig(BaseSettings):
    """Configuration settings loaded from environment variables."""

    # xAI (preferred provider)
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-02-15-preview"

    # Mistral
    mistral_api_key: str | None = None

    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Logging
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Global config instance
_config: EgileConfig | None = None


def get_config() -> EgileConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = EgileConfig()
    return _config


def set_config(config: EgileConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
