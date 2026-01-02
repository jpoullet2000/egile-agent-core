"""OpenAI LLM provider for Egile Agent Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from egile_agent_core.config import get_config
from egile_agent_core.exceptions import ProviderError
from egile_agent_core.models.base import BaseLLM, LLMResponse


@dataclass
class OpenAI(BaseLLM):
    """
    OpenAI GPT LLM provider.

    Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, and other OpenAI models.

    Example:
        ```python
        from egile_agent_core.models import OpenAI

        model = OpenAI(model="gpt-4-turbo")
        response = await model.generate([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    model: str = "gpt-4-turbo"
    temperature: float = 0.7
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    _client: AsyncOpenAI | None = field(default=None, repr=False, init=False)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            config = get_config()
            api_key = self.api_key or config.openai_api_key
            base_url = self.base_url or config.openai_base_url

            if not api_key:
                raise ProviderError(
                    "openai",
                    "API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter.",
                )

            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        return self._client

    async def generate(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """Generate a response from OpenAI."""
        client = self._get_client()

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,  # type: ignore
                "temperature": self.temperature,
                **self.extra_params,
            }
            
            if self.max_tokens:
                api_params["max_tokens"] = self.max_tokens
                
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            response = await client.chat.completions.create(**api_params)

            content = response.choices[0].message.content or ""
            
            # Extract tool calls if any
            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
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
            raise ProviderError("openai", f"Failed to generate response: {e}") from e

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        """Stream a response from OpenAI."""
        client = self._get_client()

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,  # type: ignore
                "temperature": self.temperature,
                "stream": True,
                **self.extra_params,
            }
            
            if self.max_tokens:
                api_params["max_tokens"] = self.max_tokens
                
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            stream = await client.chat.completions.create(**api_params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ProviderError("openai", f"Failed to stream response: {e}") from e
