"""Plugin base class and registry for Egile Agent Core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from egile_agent_core.agent import Agent


class Plugin(ABC):
    """
    Base class for all Egile Agent plugins.

    Plugins can extend agent functionality by hooking into the agent lifecycle.
    Implement any of the hook methods to add custom behavior.

    Example:
        ```python
        from egile_agent_core.plugins import Plugin

        class LoggingPlugin(Plugin):
            @property
            def name(self) -> str:
                return "logging"

            async def on_message_received(self, message: str) -> str:
                print(f"Received: {message}")
                return message

            async def on_response_generated(self, response: str) -> str:
                print(f"Response: {response}")
                return response
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this plugin.

        Returns:
            A unique string name for the plugin.
        """
        pass

    @property
    def description(self) -> str:
        """
        Description of what this plugin does.

        Returns:
            A human-readable description.
        """
        return ""

    @property
    def version(self) -> str:
        """
        Version of this plugin.

        Returns:
            A version string (e.g., "1.0.0").
        """
        return "0.1.0"

    async def on_agent_start(self, agent: Agent) -> None:
        """
        Called when the agent starts processing a request.

        Args:
            agent: The Agent instance that is starting.
        """
        pass

    async def on_message_received(self, message: str, **kwargs: Any) -> str:
        """
        Pre-process an incoming user message.

        Args:
            message: The original user message.
            **kwargs: Additional context.

        Returns:
            The processed message (can be modified).
        """
        return message

    async def on_response_generated(self, response: str, **kwargs: Any) -> str:
        """
        Post-process an agent response before returning to the user.

        Args:
            response: The original agent response.
            **kwargs: Additional context.

        Returns:
            The processed response (can be modified).
        """
        return response

    async def on_error(self, error: Exception, **kwargs: Any) -> None:
        """
        Called when an error occurs during agent processing.

        Args:
            error: The exception that occurred.
            **kwargs: Additional context.
        """
        pass

    def get_info(self) -> dict[str, str]:
        """Get plugin information as a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }


class PluginRegistry:
    """
    Registry for discovering and managing plugins.

    Provides a central place to register, retrieve, and list plugins.

    Example:
        ```python
        from egile_agent_core.plugins import Plugin, PluginRegistry

        class MyPlugin(Plugin):
            @property
            def name(self) -> str:
                return "my-plugin"

        # Register a plugin
        plugin = MyPlugin()
        PluginRegistry.register(plugin)

        # Get a plugin
        same_plugin = PluginRegistry.get("my-plugin")

        # List all plugins
        all_plugins = PluginRegistry.list_all()
        ```
    """

    _plugins: dict[str, Plugin] = {}

    @classmethod
    def register(cls, plugin: Plugin) -> None:
        """
        Register a plugin in the registry.

        Args:
            plugin: The plugin instance to register.

        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        if plugin.name in cls._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already registered")
        cls._plugins[plugin.name] = plugin

    @classmethod
    def unregister(cls, name: str) -> Plugin | None:
        """
        Unregister a plugin by name.

        Args:
            name: The name of the plugin to unregister.

        Returns:
            The unregistered plugin, or None if not found.
        """
        return cls._plugins.pop(name, None)

    @classmethod
    def get(cls, name: str) -> Plugin | None:
        """
        Get a plugin by name.

        Args:
            name: The name of the plugin to retrieve.

        Returns:
            The plugin instance, or None if not found.
        """
        return cls._plugins.get(name)

    @classmethod
    def list_all(cls) -> list[Plugin]:
        """
        List all registered plugins.

        Returns:
            A list of all registered plugin instances.
        """
        return list(cls._plugins.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins."""
        cls._plugins.clear()
