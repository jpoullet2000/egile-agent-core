"""FastAPI application factory for Egile Agent Core."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from egile_agent_core.server.routes import create_router
from egile_agent_core.server.agentui_routes import create_agentui_router

if TYPE_CHECKING:
    from egile_agent_core.agent import Agent


class AgentServer:
    """
    FastAPI runtime for serving agents.

    Provides a production-ready HTTP API for interacting with agents.

    Example:
        ```python
        from egile_agent_core import Agent
        from egile_agent_core.models import XAI
        from egile_agent_core.server import AgentServer

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
        agents: list[Agent],
        title: str = "Egile Agent Server",
        description: str = "AI Agent API powered by Egile Agent Core",
        version: str = "0.1.0",
        cors_origins: list[str] | None = None,
    ):
        """
        Initialize the AgentServer.

        Args:
            agents: List of Agent instances to serve.
            title: API title for OpenAPI docs.
            description: API description for OpenAPI docs.
            version: API version for OpenAPI docs.
            cors_origins: List of allowed CORS origins. Defaults to ["*"].
        """
        self.agents = {agent.name: agent for agent in agents}
        self.title = title
        self.description = description
        self.version = version
        self.cors_origins = cors_origins or ["*"]
        self._app: FastAPI | None = None

    def get_app(self) -> FastAPI:
        """
        Get or create the FastAPI application.

        Returns:
            The configured FastAPI application.
        """
        if self._app is None:
            self._app = self._create_app()
        return self._app

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Create and include original router with access to agents
        router = create_router(self.agents)
        app.include_router(router, prefix="/v1")

        # Create and include Agent UI compatible router
        agentui_router = create_agentui_router(self.agents)
        app.include_router(agentui_router)

        # Health check endpoint
        @app.get("/health")
        async def health_check() -> dict[str, str]:
            return {"status": "healthy"}

        # Root endpoint
        @app.get("/")
        async def root() -> dict[str, str]:
            return {
                "name": self.title,
                "version": self.version,
                "docs": "/docs",
            }

        return app

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
        import uvicorn

        uvicorn.run(
            self.get_app(),
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
        )
