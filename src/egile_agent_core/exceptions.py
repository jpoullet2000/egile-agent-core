"""Custom exceptions for Egile Agent Core."""


class EgileError(Exception):
    """Base exception for all Egile errors."""

    pass


class ConfigurationError(EgileError):
    """Raised when there is a configuration issue."""

    pass


class ModelError(EgileError):
    """Raised when there is an error with the LLM model."""

    pass


class ProviderError(ModelError):
    """Raised when there is an error with a specific LLM provider."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class PluginError(EgileError):
    """Raised when there is an error with a plugin."""

    def __init__(self, plugin_name: str, message: str):
        self.plugin_name = plugin_name
        super().__init__(f"Plugin '{plugin_name}': {message}")


class AgentError(EgileError):
    """Raised when there is an error with an agent."""

    pass
