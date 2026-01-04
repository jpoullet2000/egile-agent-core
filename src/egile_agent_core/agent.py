"""Core Agent class for Egile Agent Core."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from egile_agent_core.exceptions import AgentError

if TYPE_CHECKING:
    from egile_agent_core.models.base import BaseLLM
    from egile_agent_core.plugins.base import Plugin

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for LLM APIs."""
        return {"role": self.role, "content": self.content}


@dataclass
class AgentResponse:
    """Response from an agent."""

    content: str
    agent_name: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any = None


@dataclass
class Agent:
    """
    Core Agent class for interacting with LLMs.

    Example:
        ```python
        from egile_agent_core import Agent
        from egile_agent_core.models import XAI

        agent = Agent(
            name="my-agent",
            model=XAI(model="grok-4-1-fast-reasoning"),
            system_prompt="You are a helpful assistant.",
        )

        response = await agent.run("Hello, how are you?")
        print(response.content)
        ```
    """

    name: str
    model: BaseLLM
    description: str = ""
    system_prompt: str = ""
    plugins: list[Plugin] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    max_history: int = 100

    def __post_init__(self) -> None:
        """Initialize the agent after dataclass initialization."""
        # Add system message if provided
        if self.system_prompt and not self.history:
            self.history.append(Message(role="system", content=self.system_prompt))
        
        # Collect tools from plugins
        self._tools: dict[str, Any] = {}
        self._tool_schemas: list[dict[str, Any]] = []

    async def _execute_plugin_hooks(
        self, hook_name: str, value: str, **kwargs: Any
    ) -> str:
        """Execute plugin hooks in order."""
        for plugin in self.plugins:
            hook = getattr(plugin, hook_name, None)
            if hook:
                try:
                    result = await hook(value, **kwargs)
                    if result is not None:
                        value = result
                except Exception as e:
                    raise AgentError(
                        f"Plugin '{plugin.name}' hook '{hook_name}' failed: {e}"
                    ) from e
        return value

    async def _notify_plugins(self, hook_name: str, **kwargs: Any) -> None:
        """Notify plugins of an event (no return value expected)."""
        for plugin in self.plugins:
            hook = getattr(plugin, hook_name, None)
            if hook:
                try:
                    await hook(**kwargs)
                except Exception as e:
                    raise AgentError(
                        f"Plugin '{plugin.name}' hook '{hook_name}' failed: {e}"
                    ) from e

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the message list for the LLM."""
        return [msg.to_dict() for msg in self.history]
    
    def _register_tools_from_plugins(self) -> None:
        """Register tools from all plugins."""
        for plugin in self.plugins:
            # Get tools from plugin
            if hasattr(plugin, 'get_tools'):
                tools = plugin.get_tools()
                for tool_def in tools:
                    self._tool_schemas.append(tool_def)
                    # Store the plugin for later execution
                    tool_name = tool_def['function']['name']
                    self._tools[tool_name] = plugin
                    logger.info(f"Registered tool '{tool_name}' from plugin '{plugin.name}'")
    
    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments."""
        if tool_name not in self._tools:
            raise AgentError(f"Tool '{tool_name}' not found")
        
        plugin = self._tools[tool_name]
        
        # Try to get the method from the plugin
        if hasattr(plugin, tool_name):
            method = getattr(plugin, tool_name)
            try:
                result = await method(**arguments)
                return str(result)
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}")
                return f"Error: {str(e)}"
        else:
            raise AgentError(f"Tool method '{tool_name}' not found in plugin '{plugin.name}'")

    def _trim_history(self) -> None:
        """Trim history to max_history messages, preserving system message."""
        if len(self.history) <= self.max_history:
            return

        # Preserve system message if present
        system_msg = None
        if self.history and self.history[0].role == "system":
            system_msg = self.history[0]
            history = self.history[1:]
        else:
            history = self.history

        # Keep the most recent messages
        trimmed = history[-(self.max_history - (1 if system_msg else 0)) :]

        if system_msg:
            self.history = [system_msg] + trimmed
        else:
            self.history = trimmed

    async def run(self, message: str) -> AgentResponse:
        """
        Execute a single-turn agent interaction.

        Args:
            message: The user message to process.

        Returns:
            AgentResponse containing the agent's response.
        """
        # Notify plugins of agent start and register tools
        await self._notify_plugins("on_agent_start", agent=self)
        self._register_tools_from_plugins()

        # Pre-process message through plugins
        processed_message = await self._execute_plugin_hooks(
            "on_message_received", message
        )

        # Add user message to history
        self.history.append(Message(role="user", content=processed_message))

        # Generate response from LLM with tools
        messages = self._build_messages()
        tools = self._tool_schemas if self._tool_schemas else None
        
        # Tool execution loop
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            llm_response = await self.model.generate(messages, tools=tools)
            
            # Check if there are tool calls
            if not llm_response.tool_calls:
                # No tool calls, we have the final response
                response_content = llm_response.content
                break
            
            # Execute tool calls
            logger.info(f"Executing {len(llm_response.tool_calls)} tool call(s)")
            
            # Add assistant message with tool calls to history
            self.history.append(Message(
                role="assistant",
                content=llm_response.content or ""
            ))
            
            # Execute each tool and add results
            for tool_call in llm_response.tool_calls:
                function_name = tool_call['function']['name']
                try:
                    arguments = json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.info(f"Calling tool: {function_name} with args: {arguments}")
                
                # Execute the tool
                tool_result = await self._execute_tool(function_name, arguments)
                
                # Add tool result to messages for next iteration
                # Note: We're storing this in a simplified way in history
                self.history.append(Message(
                    role="user",
                    content=f"Tool '{function_name}' returned: {tool_result}"
                ))
            
            # Rebuild messages for next iteration
            messages = self._build_messages()
        
        if iteration >= max_iterations:
            response_content = "Maximum tool execution iterations reached."
            logger.warning("Maximum tool execution iterations reached")

        # Post-process response through plugins
        final_content = await self._execute_plugin_hooks(
            "on_response_generated", response_content
        )

        # Add final assistant response to history if not already added
        if not llm_response.tool_calls:
            self.history.append(Message(role="assistant", content=final_content))

        # Trim history if needed
        self._trim_history()

        return AgentResponse(
            content=final_content,
            agent_name=self.name,
            model=self.model.model_name,
            usage=llm_response.usage,
            raw_response=llm_response.raw_response,
        )

    async def stream(self, message: str) -> AsyncIterator[str]:
        """
        Stream a response from the agent.

        Args:
            message: The user message to process.

        Yields:
            String chunks of the response as they arrive.
        """
        # Notify plugins of agent start and register tools
        await self._notify_plugins("on_agent_start", agent=self)
        self._register_tools_from_plugins()

        # Pre-process message through plugins
        processed_message = await self._execute_plugin_hooks(
            "on_message_received", message
        )

        # Add user message to history
        self.history.append(Message(role="user", content=processed_message))

        # Generate response with tool support (non-streaming for tool calls)
        messages = self._build_messages()
        tools = self._tool_schemas if self._tool_schemas else None
        
        # Debug logging
        if tools:
            logger.info(f"Registered {len(tools)} tools for LLM: {[t['function']['name'] for t in tools]}")
        else:
            logger.warning("No tools registered for this agent!")
        
        # Tool execution loop
        max_iterations = 10
        iteration = 0
        full_response = ""
        
        while iteration < max_iterations:
            iteration += 1
            llm_response = await self.model.generate(messages, tools=tools)
            
            # Debug logging
            logger.info(f"LLM response - has tool_calls: {bool(llm_response.tool_calls)}, content length: {len(llm_response.content) if llm_response.content else 0}")
            
            # Check if there are tool calls
            if not llm_response.tool_calls:
                # No tool calls, stream the final response
                full_response = llm_response.content
                yield full_response
                break
            
            # Execute tool calls
            logger.info(f"Executing {len(llm_response.tool_calls)} tool call(s)")
            
            # Add assistant message with tool calls to history
            self.history.append(Message(
                role="assistant",
                content=llm_response.content or ""
            ))
            
            # Yield tool call notification
            tool_names = [tc['function']['name'] for tc in llm_response.tool_calls]
            yield f"\n🔧 Executing tools: {', '.join(tool_names)}...\n\n"
            
            # Execute each tool and add results
            for tool_call in llm_response.tool_calls:
                function_name = tool_call['function']['name']
                try:
                    arguments = json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.info(f"Calling tool: {function_name} with args: {arguments}")
                
                # Execute the tool
                tool_result = await self._execute_tool(function_name, arguments)
                
                # Yield tool result
                yield f"**Tool: {function_name}**\n{tool_result}\n\n"
                
                # Add tool result to messages for next iteration
                self.history.append(Message(
                    role="user",
                    content=f"Tool '{function_name}' returned: {tool_result}"
                ))
            
            # Rebuild messages for next iteration
            messages = self._build_messages()
        
        if iteration >= max_iterations:
            full_response = "Maximum tool execution iterations reached."
            yield full_response
            logger.warning("Maximum tool execution iterations reached")

        # Post-process full response through plugins
        final_content = await self._execute_plugin_hooks(
            "on_response_generated", full_response
        )

        # Add assistant response to history if not already added (from non-tool path)
        if not llm_response.tool_calls:
            self.history.append(Message(role="assistant", content=final_content))

        # Trim history if needed
        self._trim_history()

    def clear_history(self) -> None:
        """Clear conversation history, optionally preserving system message."""
        if self.system_prompt:
            self.history = [Message(role="system", content=self.system_prompt)]
        else:
            self.history = []

    def get_info(self) -> dict[str, Any]:
        """Get agent information as a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model.model_name,
            "provider": self.model.provider_name,
            "plugins": [p.name for p in self.plugins],
            "history_length": len(self.history),
        }
