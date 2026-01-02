"""Mistral AI LLM provider for Egile Agent Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from egile_agent_core.config import get_config
from egile_agent_core.exceptions import ProviderError
from egile_agent_core.models.base import BaseLLM, LLMResponse


@dataclass
class Mistral(BaseLLM):
    """
    Mistral AI LLM provider.

    Supports Mistral models including mistral-large, mistral-medium, and mistral-small.

    Note: Requires the 'mistral' extra to be installed:
        pip install egile-agent-core[mistral]

    Example:
        ```python
        from egile_agent_core.models import Mistral

        model = Mistral(model="mistral-large-latest")
        response = await model.generate([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    model: str = "mistral-large-latest"
    temperature: float = 0.7
    max_tokens: int | None = None
    api_key: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    _client: Any = field(default=None, repr=False, init=False)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "mistral"

    def _get_client(self) -> Any:
        """Get or create the Mistral client."""
        if self._client is None:
            try:
                from mistralai import Mistral as MistralClient
            except ImportError:
                raise ProviderError(
                    "mistral",
                    "Mistral client not installed. Install with: "
                    "pip install egile-agent-core[mistral]",
                )

            config = get_config()
            api_key = self.api_key or config.mistral_api_key

            if not api_key:
                raise ProviderError(
                    "mistral",
                    "API key not provided. Set MISTRAL_API_KEY environment variable "
                    "or pass api_key parameter.",
                )

            self._client = MistralClient(api_key=api_key)

        return self._client

    async def generate(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """Generate a response from Mistral."""
        client = self._get_client()

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                **self.extra_params,
            }
            
            if self.max_tokens:
                api_params["max_tokens"] = self.max_tokens
                
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            response = await client.chat.complete_async(**api_params)

            content = response.choices[0].message.content or ""
            
            # Extract tool calls if any
            tool_calls = []
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })
            
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content, usage=usage, raw_response=response, tool_calls=tool_calls
            )

        except Exception as e:
            raise ProviderError("mistral", f"Failed to generate response: {e}") from e

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        """Stream a response from Mistral."""
        client = self._get_client()

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                **self.extra_params,
            }
            
            if self.max_tokens:
                api_params["max_tokens"] = self.max_tokens
                
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            stream = await client.chat.stream_async(**api_params)

            async for chunk in stream:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    yield chunk.data.choices[0].delta.content

        except Exception as e:
            raise ProviderError("mistral", f"Failed to stream response: {e}") from e
