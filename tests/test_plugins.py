"""Tests for plugin system."""

import pytest

from egile_agent_core.plugins.base import Plugin, PluginRegistry
from egile_agent_core.plugins.loader import load_plugin
from egile_agent_core.exceptions import PluginError


class TestPlugin(Plugin):
    """Test plugin implementation."""

    @property
    def name(self) -> str:
        return "test-plugin"

    @property
    def description(self) -> str:
        return "A test plugin"

    @property
    def version(self) -> str:
        return "1.0.0"


class AnotherPlugin(Plugin):
    """Another test plugin."""

    @property
    def name(self) -> str:
        return "another-plugin"


class TestPluginBase:
    """Tests for Plugin base class."""

    def test_plugin_properties(self):
        plugin = TestPlugin()
        assert plugin.name == "test-plugin"
        assert plugin.description == "A test plugin"
        assert plugin.version == "1.0.0"

    def test_plugin_get_info(self):
        plugin = TestPlugin()
        info = plugin.get_info()

        assert info["name"] == "test-plugin"
        assert info["description"] == "A test plugin"
        assert info["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_plugin_hooks_have_defaults(self):
        plugin = TestPlugin()

        # Default implementations should pass through
        agent_mock = None
        await plugin.on_agent_start(agent_mock)

        result = await plugin.on_message_received("test")
        assert result == "test"

        result = await plugin.on_response_generated("response")
        assert result == "response"


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        PluginRegistry.clear()

    def test_register_plugin(self):
        plugin = TestPlugin()
        PluginRegistry.register(plugin)

        assert PluginRegistry.get("test-plugin") is plugin

    def test_register_duplicate_raises(self):
        plugin1 = TestPlugin()
        plugin2 = TestPlugin()

        PluginRegistry.register(plugin1)
        with pytest.raises(ValueError, match="already registered"):
            PluginRegistry.register(plugin2)

    def test_unregister_plugin(self):
        plugin = TestPlugin()
        PluginRegistry.register(plugin)

        removed = PluginRegistry.unregister("test-plugin")

        assert removed is plugin
        assert PluginRegistry.get("test-plugin") is None

    def test_unregister_nonexistent_returns_none(self):
        result = PluginRegistry.unregister("nonexistent")
        assert result is None

    def test_get_nonexistent_returns_none(self):
        result = PluginRegistry.get("nonexistent")
        assert result is None

    def test_list_all_plugins(self):
        plugin1 = TestPlugin()
        plugin2 = AnotherPlugin()

        PluginRegistry.register(plugin1)
        PluginRegistry.register(plugin2)

        plugins = PluginRegistry.list_all()

        assert len(plugins) == 2
        assert plugin1 in plugins
        assert plugin2 in plugins

    def test_clear_registry(self):
        plugin = TestPlugin()
        PluginRegistry.register(plugin)

        PluginRegistry.clear()

        assert len(PluginRegistry.list_all()) == 0


class TestPluginLoader:
    """Tests for plugin loader."""

    def setup_method(self):
        """Clear registry before each test."""
        PluginRegistry.clear()

    def test_load_plugin_invalid_path(self):
        with pytest.raises(PluginError):
            load_plugin("test", "invalid")

    def test_load_plugin_module_not_found(self):
        with pytest.raises(PluginError):
            load_plugin("test", "nonexistent.module:Plugin")

    def test_load_plugin_class_not_found(self):
        with pytest.raises(PluginError):
            load_plugin("test", "egile_agent_core.plugins.base:NonexistentPlugin")
