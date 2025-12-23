"""Plugin discovery and loading for Egile Agent Core."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from egile_agent_core.exceptions import PluginError
from egile_agent_core.plugins.base import Plugin, PluginRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Entry point group for plugins
PLUGIN_ENTRY_POINT_GROUP = "egile_agent_core.plugins"


def discover_plugins(register: bool = True) -> list[Plugin]:
    """
    Discover all installed plugins via entry points.

    Looks for plugins registered under the 'egile_agent_core.plugins' entry point
    group in installed packages.

    Args:
        register: If True, automatically register discovered plugins.

    Returns:
        A list of discovered Plugin instances.

    Example:
        In a plugin package's pyproject.toml:
        ```toml
        [project.entry-points."egile_agent_core.plugins"]
        my_plugin = "my_package.plugins:MyPlugin"
        ```

        Then in your code:
        ```python
        from egile_agent_core.plugins import discover_plugins

        plugins = discover_plugins()
        for plugin in plugins:
            print(f"Found plugin: {plugin.name}")
        ```
    """
    discovered: list[Plugin] = []

    # Get entry points for our plugin group
    eps = entry_points(group=PLUGIN_ENTRY_POINT_GROUP)

    for ep in eps:
        try:
            plugin = load_plugin(ep.name, ep.value, register=register)
            if plugin:
                discovered.append(plugin)
        except Exception as e:
            logger.warning(f"Failed to load plugin '{ep.name}': {e}")

    return discovered


def load_plugin(
    name: str, class_path: str, register: bool = True
) -> Plugin | None:
    """
    Load a plugin from a class path.

    Args:
        name: The entry point name of the plugin.
        class_path: The full class path (e.g., 'my_package.plugins:MyPlugin').
        register: If True, register the plugin after loading.

    Returns:
        The loaded Plugin instance, or None if loading failed.

    Raises:
        PluginError: If the plugin class is invalid or instantiation fails.
    """
    try:
        # Parse the class path
        if ":" in class_path:
            module_path, class_name = class_path.rsplit(":", 1)
        else:
            # Assume the last component is the class name
            parts = class_path.rsplit(".", 1)
            if len(parts) == 2:
                module_path, class_name = parts
            else:
                raise PluginError(name, f"Invalid class path: {class_path}")

        # Import the module
        import importlib

        module = importlib.import_module(module_path)

        # Get the class
        plugin_class = getattr(module, class_name, None)
        if plugin_class is None:
            raise PluginError(
                name, f"Class '{class_name}' not found in module '{module_path}'"
            )

        # Validate it's a Plugin subclass
        if not isinstance(plugin_class, type) or not issubclass(plugin_class, Plugin):
            raise PluginError(
                name, f"Class '{class_name}' is not a Plugin subclass"
            )

        # Instantiate the plugin
        plugin = plugin_class()

        # Register if requested
        if register:
            try:
                PluginRegistry.register(plugin)
                logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
            except ValueError as e:
                logger.warning(f"Plugin already registered: {e}")

        return plugin

    except PluginError:
        raise
    except Exception as e:
        raise PluginError(name, f"Failed to load: {e}") from e
