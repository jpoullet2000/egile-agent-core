"""xAI (Grok) LLM provider for Egile Agent Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from egile_agent_core.config import get_config
from egile_agent_core.exceptions import ProviderError
from egile_agent_core.models.base import BaseLLM, LLMResponse


@dataclass
class XAI(BaseLLM):
    """
    xAI (Grok) LLM provider.

    This is the preferred/default provider for Egile Agent Core.
    Uses the OpenAI-compatible API with xAI's base URL.

    Example:
        ```python
        from egile_agent_core.models import XAI

        model = XAI(model="grok-beta")
        response = await model.generate([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    model: str = "grok-beta"
    temperature: float = 0.7
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    _client: AsyncOpenAI | None = field(default=None, repr=False, init=False)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "xai"

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client configured for xAI."""
        if self._client is None:
            config = get_config()
            api_key = self.api_key or config.xai_api_key
            base_url = self.base_url or config.xai_base_url

            if not api_key:
                raise ProviderError(
                    "xai",
                    "API key not provided. Set XAI_API_KEY environment variable "
                    "or pass api_key parameter.",
                )

            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        return self._client

    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Generate a response from xAI."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.extra_params,
            )

            content = response.choices[0].message.content or ""
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(content=content, usage=usage, raw_response=response)

        except Exception as e:
            raise ProviderError("xai", f"Failed to generate response: {e}") from e

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream a response from xAI."""
        client = self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **self.extra_params,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ProviderError("xai", f"Failed to stream response: {e}") from e
