"""API routes for Egile Agent Server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from egile_agent_core.agent import Agent


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    message: str
    clear_history: bool = False


class ChatResponse(BaseModel):
    """Response body for chat endpoints."""

    content: str
    agent_name: str
    model: str
    usage: dict[str, int] | None = None


class AgentInfo(BaseModel):
    """Agent information response."""

    name: str
    description: str
    model: str
    provider: str
    plugins: list[str]


class AgentListResponse(BaseModel):
    """Response for listing agents."""

    agents: list[AgentInfo]


def create_router(agents: dict[str, Agent]) -> APIRouter:
    """
    Create the API router with agent-related endpoints.

    Args:
        agents: Dictionary mapping agent names to Agent instances.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(tags=["agents"])

    @router.get("/agents", response_model=AgentListResponse)
    async def list_agents() -> AgentListResponse:
        """List all available agents."""
        agent_infos = []
        for agent in agents.values():
            info = agent.get_info()
            agent_infos.append(
                AgentInfo(
                    name=info["name"],
                    description=info["description"],
                    model=info["model"],
                    provider=info["provider"],
                    plugins=info["plugins"],
                )
            )
        return AgentListResponse(agents=agent_infos)

    @router.get("/agents/{agent_name}", response_model=AgentInfo)
    async def get_agent(agent_name: str) -> AgentInfo:
        """Get details for a specific agent."""
        agent = agents.get(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_name}' not found"
            )

        info = agent.get_info()
        return AgentInfo(
            name=info["name"],
            description=info["description"],
            model=info["model"],
            provider=info["provider"],
            plugins=info["plugins"],
        )

    @router.post("/agents/{agent_name}/chat", response_model=ChatResponse)
    async def chat(agent_name: str, request: ChatRequest) -> ChatResponse:
        """
        Send a message to an agent and get a response.

        This is a non-streaming endpoint that waits for the full response.
        """
        agent = agents.get(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_name}' not found"
            )

        if request.clear_history:
            agent.clear_history()

        try:
            response = await agent.run(request.message)
            return ChatResponse(
                content=response.content,
                agent_name=response.agent_name,
                model=response.model,
                usage=response.usage,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Agent error: {str(e)}"
            ) from e

    @router.post("/agents/{agent_name}/stream")
    async def stream_chat(agent_name: str, request: ChatRequest) -> EventSourceResponse:
        """
        Send a message to an agent and stream the response.

        Returns Server-Sent Events (SSE) with response chunks.
        """
        agent = agents.get(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_name}' not found"
            )

        if request.clear_history:
            agent.clear_history()

        async def event_generator():
            try:
                async for chunk in agent.stream(request.message):
                    yield {"event": "message", "data": chunk}
                yield {"event": "done", "data": ""}
            except Exception as e:
                yield {"event": "error", "data": str(e)}

        return EventSourceResponse(event_generator())

    @router.delete("/agents/{agent_name}/history")
    async def clear_history(agent_name: str) -> dict[str, str]:
        """Clear the conversation history for an agent."""
        agent = agents.get(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_name}' not found"
            )

        agent.clear_history()
        return {"status": "ok", "message": f"History cleared for agent '{agent_name}'"}

    return router
