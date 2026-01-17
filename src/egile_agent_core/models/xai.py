"""xAI (Grok) LLM provider for Egile Agent Core."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from egile_agent_core.config import get_config
from egile_agent_core.exceptions import ProviderError
from egile_agent_core.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class XAI(BaseLLM):
    """
    xAI (Grok) LLM provider.

    This is the preferred/default provider for Egile Agent Core.
    Uses the OpenAI-compatible API with xAI's base URL.

    Example:
        ```python
        from egile_agent_core.models import XAI

        model = XAI(model="grok-4-1-fast-reasoning")
        response = await model.generate([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    model: str = "grok-4-1-fast-reasoning"
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

    async def generate(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """Generate a response from xAI."""
        client = self._get_client()
        
        # 🔍 DEBUG: Log if tools are being passed
        logger.info(f"🔍 XAI.generate called with tools: {tools is not None} ({len(tools) if tools else 0} tools)")
        if tools:
            # Handle both dict format (OpenAI-style) and function objects
            tool_names = []
            for t in tools:
                if isinstance(t, dict):
                    tool_names.append(t.get('function', {}).get('name', 'unknown'))
                elif callable(t):
                    tool_names.append(getattr(t, '__name__', 'unknown'))
                else:
                    tool_names.append(str(type(t)))
            logger.info(f"🔍 Tool names: {tool_names}")

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
                # Convert tools to OpenAI format if they're function objects
                formatted_tools = []
                for tool in tools:
                    if isinstance(tool, dict):
                        # Already in correct format
                        formatted_tools.append(tool)
                    elif callable(tool):
                        # Convert function to OpenAI format
                        # Check if tool has __tool_spec__ attribute (Agno format)
                        if hasattr(tool, '__tool_spec__'):
                            formatted_tools.append(tool.__tool_spec__)
                        else:
                            # Generate basic spec from function
                            import inspect
                            sig = inspect.signature(tool)
                            parameters = {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                            
                            for param_name, param in sig.parameters.items():
                                if param_name == 'self':
                                    continue
                                parameters["properties"][param_name] = {"type": "string"}
                                if param.default == inspect.Parameter.empty:
                                    parameters["required"].append(param_name)
                            
                            formatted_tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool.__name__.lstrip('_'),
                                    "description": tool.__doc__ or f"Function {tool.__name__}",
                                    "parameters": parameters
                                }
                            })
                    else:
                        logger.warning(f"Unknown tool format: {type(tool)}")
                
                api_params["tools"] = formatted_tools
                api_params["tool_choice"] = "auto"
                logger.info(f"🔍 Added {len(formatted_tools)} formatted tools to API params")

            response = await client.chat.completions.create(**api_params)
            
            logger.info(f"🔍 XAI response - finish_reason: {response.choices[0].finish_reason}")
            logger.info(f"🔍 Has tool_calls: {hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls is not None}")

            content = response.choices[0].message.content or ""
            
            # Extract tool calls if any
            tool_calls = []
            if response.choices[0].message.tool_calls:
                logger.info(f"🎯 XAI returned {len(response.choices[0].message.tool_calls)} tool calls!")
                for tool_call in response.choices[0].message.tool_calls:
                    logger.info(f"🎯 Tool call: {tool_call.function.name}")
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })
            else:
                logger.info(f"⚠️ XAI did NOT return tool calls, just text content")
            
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
            raise ProviderError("xai", f"Failed to generate response: {e}") from e

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        """Stream a response from xAI."""
        client = self._get_client()
        
        logger.info(f"🔍 XAI.stream called with tools: {tools is not None} ({len(tools) if tools else 0} tools)")
        if tools:
            logger.info(f"🔍 Tool names: {[t.get('function', {}).get('name') for t in tools]}")

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
                logger.info(f"🔍 Added tools to stream API params with tool_choice=auto")

            stream = await client.chat.completions.create(**api_params)
            
            tool_calls_detected = False

            async for chunk in stream:
                # Check for tool calls in the stream
                if chunk.choices and chunk.choices[0].delta.tool_calls:
                    if not tool_calls_detected:
                        logger.info(f"🎯 XAI is returning TOOL CALLS in stream!")
                        tool_calls_detected = True
                    # Tool calls are being made - we can't handle them in streaming mode
                    # The streaming API doesn't support tool execution
                    continue
                    
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            if not tool_calls_detected:
                logger.info(f"⚠️ XAI stream completed with NO tool calls")

        except Exception as e:
            raise ProviderError("xai", f"Failed to stream response: {e}") from e
