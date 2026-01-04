"""Base LLM model interface for Egile Agent Core."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    usage: dict[str, int] | None = None
    raw_response: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers must implement this interface to be compatible
    with the Agent class.
    """

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'xai')."""
        pass

    @property
    def model_name(self) -> str:
        """Return the full model name with provider."""
        return f"{self.provider_name}/{self.model}"

    @abstractmethod
    async def generate(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            tools: Optional list of tool definitions for function calling.

        Returns:
            LLMResponse containing the generated content.
        """
        pass

    @abstractmethod
    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        """
        Stream a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            tools: Optional list of tool definitions for function calling.

        Yields:
            String chunks of the response as they arrive.
        """
        pass
