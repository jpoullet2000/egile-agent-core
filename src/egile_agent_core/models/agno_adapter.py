"""Adapter to make Egile LLM models compatible with Agno's model interface."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator

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

    def __init__(self, egile_model: BaseLLM, tools: list[Callable] | None = None):
        """
        Initialize the adapter with an Egile LLM model.
        
        Args:
            egile_model: Any BaseLLM implementation from egile_agent_core.models
            tools: Optional list of callable tool functions (stored for reference)
        """
        self.egile_model = egile_model
        self._tool_map: dict[str, Callable] = {}  # Store tools by name for reference
        
        # Store tools if provided
        if tools:
            for tool in tools:
                if callable(tool):
                    # Store with both original name and stripped name (without leading _)
                    self._tool_map[tool.__name__] = tool
                    # Also store with stripped name for API compatibility
                    stripped_name = tool.__name__.lstrip('_')
                    if stripped_name != tool.__name__:
                        self._tool_map[stripped_name] = tool
        
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
    
    def _get_tools_for_api(self, tools: list[Any] | None) -> list[dict[str, Any]] | None:
        """Convert Agno tools (callables or dicts) to OpenAI API format."""
        import inspect
        
        if not tools:
            return None
        
        api_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Already in API format
                api_tools.append(tool)
            elif callable(tool):
                # Convert function to API format
                # Agno functions have __tool_spec__ attribute with OpenAI format
                if hasattr(tool, '__tool_spec__'):
                    api_tools.append(tool.__tool_spec__)
                else:
                    # Generate OpenAI function spec from function signature
                    logger.info(f"Generating tool spec for {tool.__name__}")
                    sig = inspect.signature(tool)
                    parameters = {}
                    required = []
                    
                    for param_name, param in sig.parameters.items():
                        if param_name == 'self':
                            continue
                        
                        param_spec = {"type": "string"}  # Default to string
                        
                        # Try to infer type from annotation
                        if param.annotation != inspect.Parameter.empty:
                            ann = param.annotation
                            # Handle Optional types
                            if hasattr(ann, '__origin__'):
                                if ann.__origin__ is type(None) or str(ann).startswith('typing.Optional'):
                                    # Extract the actual type from Optional[Type]
                                    args = getattr(ann, '__args__', ())
                                    if args:
                                        ann = args[0] if args[0] is not type(None) else (args[1] if len(args) > 1 else str)
                            
                            if ann == int or ann == 'int':
                                param_spec["type"] = "integer"
                            elif ann == float or ann == 'float' or param_name in ['shares', 'purchase_price', 'min_market_cap', 'max_pe', 'min_dividend_yield']:
                                param_spec["type"] = "number"
                            elif ann == bool or ann == 'bool':
                                param_spec["type"] = "boolean"
                            elif ann == str or ann == 'str':
                                param_spec["type"] = "string"
                        
                        # Infer from parameter name patterns
                        if param_spec["type"] == "string":
                            if any(keyword in param_name.lower() for keyword in ['price', 'cap', 'shares', 'yield', 'pe', 'ratio', 'min_', 'max_', 'limit']):
                                param_spec["type"] = "number"
                        
                        # Add description from docstring if available
                        if tool.__doc__:
                            param_spec["description"] = f"Parameter {param_name}"
                        
                        parameters[param_name] = param_spec
                        
                        # Mark as required if no default value
                        if param.default == inspect.Parameter.empty:
                            required.append(param_name)
                    
                    # Create OpenAI function spec
                    tool_spec = {
                        "type": "function",
                        "function": {
                            "name": tool.__name__.lstrip('_'),  # Remove leading underscore
                            "description": tool.__doc__ or f"Function {tool.__name__}",
                            "parameters": {
                                "type": "object",
                                "properties": parameters,
                                "required": required,
                            }
                        }
                    }
                    api_tools.append(tool_spec)
                    logger.info(f"Generated spec for {tool.__name__}: {tool_spec['function']['name']}")
        
        return api_tools if api_tools else None

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
            **kwargs: Additional arguments from Agno (may include tools, functions, etc.).
            
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
        
        # Extract tools from kwargs if provided by Agno
        tools = kwargs.get('tools') or kwargs.get('functions')
        
        # Log for debugging
        logger.info(f"🎬 AgnoAdapter.ainvoke called with {len(egile_messages)} messages")
        logger.info(f"🎬 kwargs keys: {list(kwargs.keys())}")
        logger.info(f"🎬 tools from kwargs: {bool(tools)} ({len(tools) if tools else 0} tools)")
        if tools:
            logger.info(f"🔍 Tool names: {[t.get('function', {}).get('name', '?') if isinstance(t, dict) else '?' for t in tools]}")
        else:
            logger.info(f"⚠️ NO TOOLS in kwargs - Agno didn't pass any!")
        
        # Call the egile model with tools if available
        logger.info(f"🎬 Calling egile_model.generate()...")
        response = await self.egile_model.generate(egile_messages, tools=tools)
        logger.info(f"🎬 generate() returned, has tool_calls: {bool(response.tool_calls)}")
        
        # Log if tool calls were made
        if response.tool_calls:
            logger.info(f"🎯 Model returned {len(response.tool_calls)} tool calls!")
            for tc in response.tool_calls:
                logger.info(f"🎯 Tool call: {tc.get('function', {}).get('name', '?')}")
        else:
            logger.info(f"⚠️ Model did NOT return tool calls")
        
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
            logger.info(f"🎬 AgnoAdapter.ainvoke_stream called!")
            logger.info(f"🎬 kwargs keys: {list(kwargs.keys())}")
            
            # Get the assistant message that Agno wants us to populate
            assistant_message = kwargs.get('assistant_message')
            accumulated_content = []
            
            # Convert Agno messages to egile format FIRST
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
            
            # Extract tools from kwargs if provided by Agno
            tools = kwargs.get('tools') or kwargs.get('functions')
            logger.info(f"🎬 tools from kwargs: {bool(tools)} ({len(tools) if tools else 0} tools)")
            
            # Log the existing tool_map from __init__
            logger.info(f"📦 Existing tool_map has {len(self._tool_map)} tools: {list(self._tool_map.keys())}")
            
            if tools:
                logger.info(f"🔍 Tool names: {[t.get('function', {}).get('name', '?') if isinstance(t, dict) else getattr(t, '__name__', '?') for t in tools]}")
                
                # Only add to tool_map if we don't already have the tools
                # DON'T reset self._tool_map - it was populated in __init__!
                for tool in tools:
                    if callable(tool):
                        # Tool is a function - use its name
                        tool_name = tool.__name__
                        if tool_name not in self._tool_map:
                            self._tool_map[tool_name] = tool
                            logger.info(f"📦 Added callable tool: {tool_name}")
                    elif isinstance(tool, dict):
                        # Tool is a dict spec - we already have the callable from __init__
                        tool_name = tool.get('function', {}).get('name')
                        if tool_name and tool_name not in self._tool_map:
                            logger.warning(f"⚠️ Tool {tool_name} is dict spec and not in tool_map")
                
                logger.info(f"📦 Final tool_map has {len(self._tool_map)} callable tools: {list(self._tool_map.keys())}")
            else:
                logger.info(f"⚠️ NO TOOLS in ainvoke_stream kwargs!")
            
            # CRITICAL: When tools are present, use generate() not stream()
            # Streaming doesn't properly handle tool calls - XAI returns tool_calls in the response
            # but streaming only yields text chunks
            if tools:
                logger.info(f"🔄 SWITCHING to generate() mode because tools are present")
                # Convert tools to API format if needed (handles both callables and dicts)
                api_tools = self._get_tools_for_api(tools)
                logger.info(f"🔧 Converted to {len(api_tools) if api_tools else 0} API format tools")
                
                # Tool execution loop - handle multiple rounds of tool calls
                max_iterations = 5
                iteration = 0
                response = None
                
                while iteration < max_iterations:
                    iteration += 1
                    logger.info(f"🔄 Tool execution iteration {iteration}/{max_iterations}")
                    
                    # Call model (with or without tools depending on iteration)
                    should_allow_tools = iteration < max_iterations
                    response = await self.egile_model.generate(
                        egile_messages, 
                        tools=api_tools if should_allow_tools else None
                    )
                    
                    # Check if model returned tool calls
                    if response.tool_calls:
                        logger.info(f"🎯 Model returned {len(response.tool_calls)} tool calls!")
                        for tc in response.tool_calls:
                            logger.info(f"🎯   - {tc.get('function', {}).get('name', '?')}")
                        
                        # Set tool calls on the assistant message
                        if assistant_message is not None:
                            assistant_message.content = response.content or ""
                            assistant_message.tool_calls = response.tool_calls
                            logger.info(f"✅ Set {len(response.tool_calls)} tool_calls on assistant_message")
                        
                        # Check if ALL tool calls are in our tool_map
                        # If ANY tool is NOT in our map, it's an Agno-managed tool (like delegate_task_to_member)
                        # In that case, return the tool calls to Agno and let Agno execute them
                        all_tools_in_map = all(
                            tc['function']['name'] in self._tool_map 
                            for tc in response.tool_calls
                        )
                        
                        if not all_tools_in_map:
                            logger.info(f"🔄 Some tools are Agno-managed (not in tool_map) - returning to Agno for execution")
                            # Just return - Agno will see the tool_calls on assistant_message and execute them
                            return
                        
                        # All tools are in our map - execute them ourselves
                        logger.info(f"🔧 Executing {len(response.tool_calls)} tool calls...")
                        
                        # Check if we're calling the same tool consecutively - prevent infinite loops
                        last_tool_call = getattr(self, '_last_tool_call', None)
                        last_tool_failed = getattr(self, '_last_tool_failed', False)
                        current_tool_name = response.tool_calls[0]['function']['name'] if response.tool_calls else None
                        
                        # Prevent retrying a tool that just failed OR calling get_last_draft consecutively
                        if last_tool_call == current_tool_name and (last_tool_failed or current_tool_name in ['get_last_draft', 'get_draft']):
                            logger.warning(f"⚠️ Preventing consecutive calls to {current_tool_name} (failed={last_tool_failed}) - returning error")
                            tool_results = [{
                                "tool_call_id": response.tool_calls[0]['id'],
                                "role": "tool",
                                "name": current_tool_name,
                                "content": f"Error: Tool {current_tool_name} {'already failed' if last_tool_failed else 'cannot be called twice in a row'}. Stop trying to use this tool and respond to the user explaining the issue."
                            }]
                            # Reset the failed flag to allow trying again later (but not consecutively)
                            self._last_tool_failed = False
                        else:
                            self._last_tool_call = current_tool_name
                            self._last_tool_failed = False  # Reset at start of execution
                            tool_results = []
                            for tc in response.tool_calls:
                                tool_name = tc['function']['name']
                                tool_args_str = tc['function']['arguments']
                                tool_id = tc['id']
                                
                                try:
                                    # Parse arguments
                                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                                    logger.info(f"🔧 Calling {tool_name}({tool_args})")
                                    
                                    # Execute tool
                                    tool_func = self._tool_map[tool_name]
                                    import inspect
                                    if inspect.iscoroutinefunction(tool_func):
                                        result = await tool_func(**tool_args)
                                    else:
                                        result = tool_func(**tool_args)
                                    
                                    logger.info(f"✅ Tool {tool_name} returned: {str(result)[:100]}")
                                    tool_results.append({
                                        "tool_call_id": tool_id,
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": str(result)
                                    })
                                except Exception as e:
                                    logger.error(f"❌ Tool {tool_name} failed: {e}")
                                    self._last_tool_failed = True  # Mark this tool as failed
                                    tool_results.append({
                                        "tool_call_id": tool_id,
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": f"Error: {str(e)}"
                                    })
                        
                        # Add tool results to messages for next iteration
                        logger.info(f"🔄 Adding {len(tool_results)} tool results to conversation")
                        for result in tool_results:
                            egile_messages.append(result)
                        
                        # Continue to next iteration
                        continue
                    
                    # No more tool calls - we have final response
                    logger.info(f"✅ Model returned final response (no tool calls)")
                    break
                
                # After loop, yield the final response
                if response and response.content:
                    logger.info(f"📝 Yielding final response: {len(response.content)} characters")
                    for chunk in response.content.split():
                        yield ModelResponse(content=chunk + " ", role="assistant")
                
                if assistant_message is not None and response:
                    assistant_message.content = response.content
                
                return
            
            logger.debug(f"Converted to egile messages: {egile_messages}")
            logger.debug(f"Assistant message before streaming: {assistant_message}")
            
            # Stream from the egile model WITH TOOLS
            logger.info(f"🎬 Calling egile_model.stream() with {len(tools) if tools else 0} tools...")
            async for chunk in self.egile_model.stream(egile_messages, tools=tools):
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
