"""Plugin system for Egile Agent Core."""

from egile_agent_core.plugins.base import Plugin, PluginRegistry
from egile_agent_core.plugins.loader import discover_plugins, load_plugin

__all__ = [
    "Plugin",
    "PluginRegistry",
    "discover_plugins",
    "load_plugin",
]
