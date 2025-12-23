"""Agent UI compatible API routes for Egile Agent Server."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from egile_agent_core.agent import Agent


# ============= Request/Response Models =============


class AgentModel(BaseModel):
    """Model information for an agent."""

    name: str
    model: str
    provider: str


class AgentDetails(BaseModel):
    """Agent details response matching Agent UI expectations."""

    id: str
    name: str
    description: str | None = None
    model: AgentModel
    db_id: str | None = None


class RunRequest(BaseModel):
    """Request to run an agent (start a conversation)."""

    message: str
    stream: bool = True
    session_id: str | None = None
    user_id: str | None = None


class RunEvent(BaseModel):
    """Server-Sent Event for streaming responses."""

    event: str
    content: str | dict | None = None
    run_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"


# ============= Route Creation =============


def create_agentui_router(agents: dict[str, Agent]) -> APIRouter:
    """
    Create Agent UI compatible API router.

    This router provides endpoints that match the Agent UI's expected API format,
    allowing your egile-agent-core backend to work seamlessly with the Agent UI frontend.

    Args:
        agents: Dictionary mapping agent names to Agent instances.

    Returns:
        Configured APIRouter for Agent UI integration.
    """
    router = APIRouter(tags=["agentui"])

    # Store sessions in memory (in production, use a proper database)
    sessions: dict[str, list[dict[str, Any]]] = {}

    @router.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", version="0.1.0")

    @router.get("/agents", response_model=list[AgentDetails])
    async def list_agents() -> list[AgentDetails]:
        """
        List all available agents in Agent UI format.

        This endpoint returns agent information in the format expected by Agent UI.
        """
        agent_details = []
        for agent_name, agent in agents.items():
            info = agent.get_info()

            # Create a unique agent_id based on the name
            agent_id = agent_name.replace(" ", "_").lower()

            agent_details.append(
                AgentDetails(
                    id=agent_id,
                    name=info["name"],
                    description=info["description"],
                    model=AgentModel(
                        name=info["model"],
                        model=info["model"],
                        provider=info["provider"],
                    ),
                )
            )
        return agent_details

    @router.post("/agents/{agent_id}/runs")
    async def run_agent(
        agent_id: str,
        message: str = Form(...),
        stream: str = Form("true"),
        session_id: str | None = Form(None),
        user_id: str | None = Form(None),
    ) -> EventSourceResponse:
        """
        Run an agent and stream the response in Agent UI format.

        This endpoint streams Server-Sent Events (SSE) compatible with Agent UI,
        including RunStarted, RunContent, and RunCompleted events.

        Args:
            agent_id: The unique identifier for the agent
            message: The user's message (from FormData)
            stream: Whether to stream the response (from FormData)
            session_id: Optional session ID (from FormData)
            user_id: Optional user ID (from FormData)

        Returns:
            EventSourceResponse with streaming events
        """
        # Find the agent by ID
        agent = None
        for agent_name, a in agents.items():
            if agent_name.replace(" ", "_").lower() == agent_id:
                agent = a
                break

        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_id}' not found"
            )

        # Generate IDs
        run_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())

        # Store message in session
        if session_id not in sessions:
            sessions[session_id] = []

        sessions[session_id].append(
            {
                "role": "user",
                "content": message,
                "created_at": int(datetime.now().timestamp() * 1000),
            }
        )

        async def event_generator():
            """Generate SSE events in Agent UI format."""
            try:
                # Send RunStarted event
                yield {
                    "event": "RunStarted",
                    "data": json.dumps(
                        {
                            "event": "RunStarted",
                            "run_id": run_id,
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "created_at": int(datetime.now().timestamp() * 1000),
                        }
                    ),
                }

                # Stream the agent response
                full_content = ""
                async for chunk in agent.stream(message):
                    full_content += chunk

                    # Send RunContent event for each chunk
                    yield {
                        "event": "RunContent",
                        "data": json.dumps(
                            {
                                "event": "RunContent",
                                "content": chunk,
                                "content_type": "text",
                                "run_id": run_id,
                                "session_id": session_id,
                                "agent_id": agent_id,
                                "created_at": int(datetime.now().timestamp() * 1000),
                            }
                        ),
                    }

                # Store assistant response in session
                sessions[session_id].append(
                    {
                        "role": "assistant",
                        "content": full_content,
                        "created_at": int(datetime.now().timestamp() * 1000),
                    }
                )

                # Send RunCompleted event
                yield {
                    "event": "RunCompleted",
                    "data": json.dumps(
                        {
                            "event": "RunCompleted",
                            "content": full_content,
                            "content_type": "text",
                            "run_id": run_id,
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "created_at": int(datetime.now().timestamp() * 1000),
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": full_content,
                                    "created_at": int(datetime.now().timestamp() * 1000),
                                    "name": agent.name,
                                    "tool_call_id": None,
                                    "tool_calls": None,
                                }
                            ],
                        }
                    ),
                }

            except Exception as e:
                # Send RunError event
                yield {
                    "event": "RunError",
                    "data": json.dumps(
                        {
                            "event": "RunError",
                            "content": str(e),
                            "run_id": run_id,
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "created_at": int(datetime.now().timestamp() * 1000),
                        }
                    ),
                }

        return EventSourceResponse(event_generator())

    @router.get("/sessions")
    async def get_sessions(
        type: str = "agent",
        component_id: str | None = None,
        db_id: str | None = None,
    ) -> dict[str, list]:
        """
        Get all sessions (conversation history).

        Note: This is a basic implementation. In production, use a proper database.
        """
        # Return empty list for now
        # This can be enhanced to return actual session data
        return {"data": []}

    @router.get("/sessions/{session_id}/runs")
    async def get_session_runs(
        session_id: str,
        type: str = "agent",
        db_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get runs for a specific session.

        Returns the conversation history for a session.
        """
        if session_id not in sessions:
            return {"data": []}

        return {
            "data": [
                {
                    "run_id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "messages": sessions[session_id],
                }
            ]
        }

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, db_id: str | None = None) -> dict[str, str]:
        """Delete a session and its history."""
        if session_id in sessions:
            del sessions[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}

    @router.get("/teams")
    async def list_teams() -> list:
        """
        List all available teams.
        
        Agent UI expects this endpoint. For now, we return an empty list
        since egile-agent-core uses individual agents, not teams.
        """
        return []

    return router
