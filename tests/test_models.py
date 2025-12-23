"""Tests for LLM model providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from egile_agent_core.models.base import BaseLLM, LLMResponse
from egile_agent_core.models.xai import XAI
from egile_agent_core.models.openai import OpenAI
from egile_agent_core.models.azure_openai import AzureOpenAI
from egile_agent_core.exceptions import ProviderError


class TestBaseLLM:
    """Tests for BaseLLM interface."""

    def test_base_llm_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLLM()


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_response_creation(self):
        response = LLMResponse(
            content="Hello",
            usage={"total_tokens": 10},
        )
        assert response.content == "Hello"
        assert response.usage == {"total_tokens": 10}

    def test_response_optional_fields(self):
        response = LLMResponse(content="Hello")
        assert response.usage is None
        assert response.raw_response is None


class TestXAI:
    """Tests for xAI provider."""

    def test_xai_defaults(self):
        model = XAI()
        assert model.model == "grok-beta"
        assert model.provider_name == "xai"
        assert model.model_name == "xai/grok-beta"

    def test_xai_custom_model(self):
        model = XAI(model="grok-2")
        assert model.model == "grok-2"
        assert model.model_name == "xai/grok-2"

    def test_xai_requires_api_key(self):
        model = XAI()
        with pytest.raises(ProviderError) as exc_info:
            model._get_client()
        assert "API key not provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_xai_generate_with_mock(self):
        with patch("egile_agent_core.models.xai.get_config") as mock_config:
            mock_config.return_value.xai_api_key = "test-key"
            mock_config.return_value.xai_base_url = "https://api.x.ai/v1"

            model = XAI()

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30

            with patch.object(model, "_client") as mock_client:
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                model._client = mock_client

                response = await model.generate([{"role": "user", "content": "Hello"}])

                assert response.content == "Test response"
                assert response.usage["total_tokens"] == 30


class TestOpenAI:
    """Tests for OpenAI provider."""

    def test_openai_defaults(self):
        model = OpenAI()
        assert model.model == "gpt-4-turbo"
        assert model.provider_name == "openai"

    def test_openai_requires_api_key(self):
        model = OpenAI()
        with pytest.raises(ProviderError) as exc_info:
            model._get_client()
        assert "API key not provided" in str(exc_info.value)


class TestAzureOpenAI:
    """Tests for Azure OpenAI provider."""

    def test_azure_defaults(self):
        model = AzureOpenAI()
        assert model.model == "gpt-4"
        assert model.provider_name == "azure-openai"

    def test_azure_requires_api_key(self):
        model = AzureOpenAI()
        with pytest.raises(ProviderError) as exc_info:
            model._get_client()
        assert "API key not provided" in str(exc_info.value)

    def test_azure_requires_endpoint(self):
        model = AzureOpenAI(api_key="test-key")
        with pytest.raises(ProviderError) as exc_info:
            model._get_client()
        assert "Endpoint not provided" in str(exc_info.value)
