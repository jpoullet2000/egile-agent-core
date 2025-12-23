"""Core Agent class for Egile Agent Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from egile_agent_core.exceptions import AgentError

if TYPE_CHECKING:
    from egile_agent_core.models.base import BaseLLM
    from egile_agent_core.plugins.base import Plugin


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
            model=XAI(model="grok-beta"),
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
        # Notify plugins of agent start
        await self._notify_plugins("on_agent_start", agent=self)

        # Pre-process message through plugins
        processed_message = await self._execute_plugin_hooks(
            "on_message_received", message
        )

        # Add user message to history
        self.history.append(Message(role="user", content=processed_message))

        # Generate response from LLM
        messages = self._build_messages()
        llm_response = await self.model.generate(messages)

        # Extract content from response
        response_content = llm_response.content

        # Post-process response through plugins
        final_content = await self._execute_plugin_hooks(
            "on_response_generated", response_content
        )

        # Add assistant response to history
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
        # Notify plugins of agent start
        await self._notify_plugins("on_agent_start", agent=self)

        # Pre-process message through plugins
        processed_message = await self._execute_plugin_hooks(
            "on_message_received", message
        )

        # Add user message to history
        self.history.append(Message(role="user", content=processed_message))

        # Stream response from LLM
        messages = self._build_messages()
        full_response = ""

        async for chunk in self.model.stream(messages):
            full_response += chunk
            yield chunk

        # Post-process full response through plugins
        final_content = await self._execute_plugin_hooks(
            "on_response_generated", full_response
        )

        # Add assistant response to history
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
