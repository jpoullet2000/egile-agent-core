"""Adapter to make Egile LLM models compatible with Agno's model interface."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from agno.models.base import Model, Message
from agno.models.response import ModelResponse

if TYPE_CHECKING:
    from egile_agent_core.models.base import BaseLLM

logger = logging.getLogger(__name__)


class AgnoModelAdapter(Model):
    """
    Adapter that wraps Egile LLM models to work with Agno's Agent framework.
    
    This allows using egile-agent-core's multi-provider LLM support (XAI, OpenAI, 
    Azure, Mistral) within Agno's AgentOS.
    
    Example:
        ```python
        from egile_agent_core.models import XAI
        from egile_agent_core.models.agno_adapter import AgnoModelAdapter
        from agno.agent import Agent
        
        egile_model = XAI(model="grok-4-1-fast-reasoning")
        agno_model = AgnoModelAdapter(egile_model)
        
        agent = Agent(
            name="my-agent",
            model=agno_model,
            instructions=["You are a helpful assistant."],
        )
        ```
    """

    def __init__(self, egile_model: BaseLLM):
        """
        Initialize the adapter with an Egile LLM model.
        
        Args:
            egile_model: Any BaseLLM implementation from egile_agent_core.models
        """
        self.egile_model = egile_model
        super().__init__(
            id=egile_model.model,
            name=egile_model.model_name,
            provider=egile_model.provider_name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return model configuration as dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "egile_model": self.egile_model.model,
            "temperature": self.egile_model.temperature,
        }

    def invoke(self, messages: list[Message]) -> str:
        """
        Synchronous invoke - not used, raises NotImplementedError.
        Agno prefers async methods.
        """
        raise NotImplementedError("Use ainvoke instead")

    def invoke_stream(self, messages: list[Message]) -> Iterator[ModelResponse]:
        """
        Synchronous stream - not used, raises NotImplementedError.
        Agno prefers async methods.
        """
        raise NotImplementedError("Use ainvoke_stream instead")

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> str:
        """
        Invoke the model with messages and return the response.
        
        Args:
            messages: List of Agno Message objects.
            **kwargs: Additional arguments from Agno (ignored by egile models).
            
        Returns:
            The assistant's response content.
        """
        # Convert Agno messages to egile format
        egile_messages = []
        
        # Handle different message formats
        if not isinstance(messages, list):
            messages = [messages]
            
        for msg in messages:
            if isinstance(msg, str):
                # String message - treat as user message
                egile_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                # Already a dictionary
                egile_messages.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # Message object with role and content attributes
                egile_messages.append({"role": msg.role, "content": msg.content})
            elif hasattr(msg, 'model_dump'):
                # Pydantic model
                egile_messages.append(msg.model_dump())
            else:
                # Fallback - convert to string and treat as user message
                egile_messages.append({"role": "user", "content": str(msg)})
        
        # Call the egile model
        response = await self.egile_model.generate(egile_messages)
        return response.content

    async def ainvoke_stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:
        """
        Stream the model response.
        
        Args:
            messages: List of Agno Message objects.
            **kwargs: Additional arguments from Agno. May include:
                - assistant_message: Message object to populate with streaming content
            
        Yields:
            ModelResponse objects containing string chunks of the response.
        """
        try:
            logger.debug(f"ainvoke_stream called with messages type: {type(messages)}")
            logger.debug(f"Messages: {messages}")
            logger.debug(f"Kwargs: {kwargs}")
            
            # Get the assistant message that Agno wants us to populate
            assistant_message = kwargs.get('assistant_message')
            accumulated_content = []
            
            # Convert Agno messages to egile format
            egile_messages = []
            
            # Handle different message formats
            if not isinstance(messages, list):
                messages = [messages]
                
            for i, msg in enumerate(messages):
                logger.debug(f"Processing message {i}: type={type(msg)}, value={msg}")
                
                if isinstance(msg, str):
                    # String message - treat as user message
                    egile_messages.append({"role": "user", "content": msg})
                elif isinstance(msg, dict):
                    # Already a dictionary
                    egile_messages.append(msg)
                elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # Message object with role and content attributes
                    egile_messages.append({"role": msg.role, "content": msg.content})
                elif hasattr(msg, 'model_dump'):
                    # Pydantic model
                    egile_messages.append(msg.model_dump())
                else:
                    # Fallback - convert to string and treat as user message
                    egile_messages.append({"role": "user", "content": str(msg)})
            
            logger.debug(f"Converted to egile messages: {egile_messages}")
            logger.debug(f"Assistant message before streaming: {assistant_message}")
            
            # Stream from the egile model
            async for chunk in self.egile_model.stream(egile_messages):
                # Accumulate content if we have an assistant message to populate
                if assistant_message is not None:
                    accumulated_content.append(chunk)
                
                # Yield ModelResponse object instead of raw string
                yield ModelResponse(content=chunk, role="assistant")
            
            # After streaming completes, populate the assistant message with accumulated content
            # Agno expects us to set the content field with the complete response
            if assistant_message is not None and accumulated_content:
                final_content = "".join(accumulated_content)
                logger.debug(f"Setting assistant_message.content to: {final_content[:100]}...")
                assistant_message.content = final_content
                logger.debug(f"Assistant message after streaming: role={assistant_message.role}, content set={assistant_message.content is not None}")
                
        except Exception as e:
            logger.error(f"Error in ainvoke_stream: {e}", exc_info=True)
            raise

    def _parse_provider_response(self, response: Any) -> ModelResponse:
        """
        Parse provider response - converts response to ModelResponse.
        This is used by Agno's framework.
        """
        if isinstance(response, ModelResponse):
            return response
        # Convert string or other response to ModelResponse
        return ModelResponse(content=str(response), role="assistant")

    def _parse_provider_response_delta(self, delta: Any) -> ModelResponse:
        """
        Parse provider response delta for streaming - converts delta to ModelResponse.
        This is used by Agno's framework for streaming responses.
        """
        if isinstance(delta, ModelResponse):
            return delta
        # Convert string or other delta to ModelResponse
        return ModelResponse(content=str(delta), role="assistant")
        return str(delta)
