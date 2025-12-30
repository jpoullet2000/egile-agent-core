"""AgentOS server runtime for Egile Agent Core."""

from egile_agent_core.server.app import AgentServer, create_agent_os

__all__ = ["AgentServer", "create_agent_os"]


def main() -> None:
    """CLI entry point for starting the server."""
    import argparse

    from egile_agent_core import Agent
    from egile_agent_core.config import get_config
    from egile_agent_core.models import XAI

    parser = argparse.ArgumentParser(description="Egile Agent Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    config = get_config()

    # Create a default agent
    default_agent = Agent(
        name="default",
        model=XAI(model="grok-4-1-fast-reasoning"),
        description="Default Egile agent powered by xAI Grok",
        system_prompt="You are a helpful AI assistant.",
    )

    # Create and run server
    server = AgentServer(agents=[default_agent])
    server.serve(
        host=args.host or config.server_host,
        port=args.port or config.server_port,
        reload=args.reload,
    )
