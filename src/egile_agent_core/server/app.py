"""AgentOS server for Egile Agent Core.

This module provides integration with Agno's AgentOS framework, allowing egile-agent-core
to leverage AgentOS features like automatic API routing, session management, and Agent UI
compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agno.agent import Agent as AgnoAgent
from agno.db.sqlite import AsyncSqliteDb
from agno.os import AgentOS

from egile_agent_core.models.agno_adapter import AgnoModelAdapter

if TYPE_CHECKING:
    from egile_agent_core.models.base import BaseLLM


def create_agent_os(
    agents_config: list[dict],
    os_id: str = "egile-agent-os",
    description: str = "AI Agent OS powered by Egile Agent Core",
    db_file: str = "agent_os.db",
) -> AgentOS:
    """
    Create an AgentOS instance with egile-agent-core models.

    Example:
        ```python
        from egile_agent_core.models import XAI
        from egile_agent_core.server import create_agent_os

        agents_config = [
            {
                "name": "my-agent",
                "model": XAI(model="grok-4-1-fast-reasoning"),
                "instructions": ["You are a helpful assistant."],
                "description": "A helpful AI assistant",
            }
        ]

        agent_os = create_agent_os(agents_config)
        agent_os.serve(port=8000, reload=True)
        ```

    Args:
        agents_config: List of agent configuration dictionaries. Each dict should have:
            - name: Agent name
            - model: BaseLLM instance (XAI, OpenAI, etc.)
            - instructions: List of instruction strings
            - description: Optional agent description
            - plugins: Optional list of egile plugin instances
        os_id: Unique ID for this AgentOS instance
        description: Description of the AgentOS
        db_file: Path to SQLite database file for session storage

    Returns:
        Configured AgentOS instance ready to serve
    """
    # Create shared database for all agents
    db = AsyncSqliteDb(db_file=db_file)

    # Convert egile agents to Agno agents
    agno_agents = []
    for config in agents_config:
        egile_model: BaseLLM = config["model"]
        
        # Convert egile plugins to Agno tools
        tools = []
        plugins = config.get("plugins", [])
        for plugin in plugins:
            if hasattr(plugin, "get_tool_functions"):
                # Get tool functions from plugin
                tool_functions = plugin.get_tool_functions()
                for func_name, func in tool_functions.items():
                    tools.append(func)
        
        # Log registered tools
        import logging
        logger = logging.getLogger(__name__)
        if tools:
            logger.info(f"Registered {len(tools)} tools for agent '{config['name']}': {[t.__name__ for t in tools]}")
        else:
            logger.warning(f"No tools registered for agent '{config['name']}'")
        
        # Create adapter with tools
        agno_model = AgnoModelAdapter(egile_model, tools=tools)

        agent = AgnoAgent(
            name=config["name"],
            model=agno_model,
            db=db,
            instructions=config.get("instructions", []),
            description=config.get("description", ""),
            tools=tools,
            markdown=config.get("markdown", True),
            debug_mode=config.get("debug_mode", False),
        )
        agno_agents.append(agent)

    # Create and return AgentOS
    agent_os = AgentOS(
        id=os_id,
        description=description,
        agents=agno_agents,
    )

    return agent_os


class AgentServer:
    """
    Compatibility wrapper for existing AgentServer API.

    This maintains backward compatibility while using AgentOS under the hood.
    For new code, prefer using create_agent_os() directly.

    Example:
        ```python
        from egile_agent_core import Agent
        from egile_agent_core.models import XAI
        from egile_agent_core.server import AgentServer

        # This still works for backward compatibility
        agent = Agent(
            name="my-agent",
            model=XAI(model="grok-4-1-fast-reasoning"),
            system_prompt="You are a helpful assistant.",
        )

        server = AgentServer(agents=[agent])
        server.serve(host="0.0.0.0", port=8000)
        ```
    """

    def __init__(
        self,
        agents: list,  # List of egile_agent_core.agent.Agent instances
        title: str = "Egile Agent Server",
        description: str = "AI Agent API powered by Egile Agent Core",
        version: str = "0.1.0",
        cors_origins: list[str] | None = None,
        db_file: str = "agent_os.db",
    ):
        """
        Initialize the AgentServer (compatibility mode).

        Args:
            agents: List of egile_agent_core.agent.Agent instances.
            title: API title (used as OS description).
            description: API description.
            version: API version.
            cors_origins: Not used in AgentOS (kept for compatibility).
            db_file: Path to SQLite database file.
        """
        # Convert egile Agent instances to AgentOS config format
        agents_config = []
        for agent in agents:
            agents_config.append({
                "name": agent.name,
                "model": agent.model,
                "instructions": [agent.system_prompt] if agent.system_prompt else [],
                "description": agent.description,
            })

        self.agent_os = create_agent_os(
            agents_config=agents_config,
            os_id=title.lower().replace(" ", "-"),
            description=description,
            db_file=db_file,
        )
        self.title = title
        self.description = description
        self.version = version

    def get_app(self):
        """
        Get the FastAPI application.

        Returns:
            The AgentOS FastAPI application.
        """
        return self.agent_os.get_app()

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info",
    ) -> None:
        """
        Start the server using uvicorn.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            reload: Enable auto-reload for development.
            log_level: Uvicorn log level.
        """
        # Note: AgentOS.serve() doesn't support all these params directly
        # We'll need to call it with what it supports
        self.agent_os.serve(
            reload=reload,
            port=port,
        )
