"""Azure OpenAI LLM provider for Egile Agent Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncAzureOpenAI

from egile_agent_core.config import get_config
from egile_agent_core.exceptions import ProviderError
from egile_agent_core.models.base import BaseLLM, LLMResponse


@dataclass
class AzureOpenAI(BaseLLM):
    """
    Azure OpenAI LLM provider.

    Connects to Azure-hosted OpenAI models using Azure-specific configuration.

    Example:
        ```python
        from egile_agent_core.models import AzureOpenAI

        model = AzureOpenAI(
            model="gpt-4",  # The deployment name in Azure
        )
        response = await model.generate([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    model: str = "gpt-4"  # Azure deployment name
    temperature: float = 0.7
    max_tokens: int | None = None
    api_key: str | None = None
    endpoint: str | None = None
    api_version: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    _client: AsyncAzureOpenAI | None = field(default=None, repr=False, init=False)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "azure-openai"

    def _get_client(self) -> AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client."""
        if self._client is None:
            config = get_config()
            api_key = self.api_key or config.azure_openai_api_key
            endpoint = self.endpoint or config.azure_openai_endpoint
            api_version = self.api_version or config.azure_openai_api_version

            if not api_key:
                raise ProviderError(
                    "azure-openai",
                    "API key not provided. Set AZURE_OPENAI_API_KEY environment "
                    "variable or pass api_key parameter.",
                )

            if not endpoint:
                raise ProviderError(
                    "azure-openai",
                    "Endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment "
                    "variable or pass endpoint parameter.",
                )

            self._client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )

        return self._client

    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Generate a response from Azure OpenAI."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self.model,  # This is the deployment name in Azure
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
            raise ProviderError(
                "azure-openai", f"Failed to generate response: {e}"
            ) from e

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream a response from Azure OpenAI."""
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
            raise ProviderError(
                "azure-openai", f"Failed to stream response: {e}"
            ) from e
