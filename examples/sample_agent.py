"""
Sample Egile Agent Core Application

This example demonstrates how to create a simple agent and run it with the server.

Usage:
    python -m examples.sample_agent

Or with environment variables:
    XAI_API_KEY=your-key python -m examples.sample_agent
"""

import asyncio
import os

from egile_agent_core import Agent
from egile_agent_core.models import XAI, OpenAI
from egile_agent_core.plugins import Plugin
from egile_agent_core.server import AgentServer


# Example: Custom logging plugin
class LoggingPlugin(Plugin):
    """A simple plugin that logs all messages and responses."""

    @property
    def name(self) -> str:
        return "logging"

    @property
    def description(self) -> str:
        return "Logs all messages and responses for debugging"

    async def on_message_received(self, message: str, **kwargs) -> str:
        print(f"📥 User: {message}")
        return message

    async def on_response_generated(self, response: str, **kwargs) -> str:
        print(f"📤 Agent: {response[:100]}..." if len(response) > 100 else f"📤 Agent: {response}")
        return response


async def demo_single_turn():
    """Demonstrate single-turn agent interaction."""
    print("\n" + "=" * 50)
    print("Single-Turn Demo")
    print("=" * 50)

    # Create agent with xAI (preferred)
    agent = Agent(
        name="demo-agent",
        model=XAI(model="grok-4-1-fast-reasoning"),
        system_prompt="You are a helpful assistant. Be concise.",
        plugins=[LoggingPlugin()],
    )

    # Run a single interaction
    response = await agent.run("What is Python in one sentence?")
    print(f"\nFull Response:\n{response.content}")
    print(f"\nModel: {response.model}")
    if response.usage:
        print(f"Tokens: {response.usage}")


async def demo_streaming():
    """Demonstrate streaming agent interaction."""
    print("\n" + "=" * 50)
    print("Streaming Demo")
    print("=" * 50)

    agent = Agent(
        name="stream-agent",
        model=XAI(model="grok-4-1-fast-reasoning"),
        system_prompt="You are a helpful assistant.",
    )

    print("\n📥 User: Tell me a short joke")
    print("📤 Agent: ", end="", flush=True)

    async for chunk in agent.stream("Tell me a short joke"):
        print(chunk, end="", flush=True)

    print("\n")


def run_server():
    """Run the agent server."""
    print("\n" + "=" * 50)
    print("Starting Agent Server")
    print("=" * 50)

    # Create multiple agents
    agents = [
        Agent(
            name="grok",
            model=XAI(model="grok-4-1-fast-reasoning"),
            description="General purpose assistant powered by xAI Grok",
            system_prompt="You are Grok, a helpful AI assistant. Be witty and informative.",
        ),
    ]

    # Only add OpenAI agent if key is available
    if os.getenv("OPENAI_API_KEY"):
        agents.append(
            Agent(
                name="gpt",
                model=OpenAI(model="gpt-4-turbo"),
                description="General purpose assistant powered by OpenAI GPT-4",
                system_prompt="You are a helpful AI assistant powered by GPT-4.",
            )
        )

    # Create and run server
    server = AgentServer(
        agents=agents,
        title="Egile Agent Demo Server",
        description="Demo server showcasing Egile Agent Core capabilities",
    )

    print(f"\n🚀 Starting server with {len(agents)} agent(s)")
    print("   - Visit http://localhost:8000/docs for API documentation")
    print("   - Visit http://localhost:8000/v1/agents to list agents")
    print("\nPress Ctrl+C to stop...\n")

    server.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    else:
        # Run demos
        print("Egile Agent Core - Demo")
        print("========================")

        if not os.getenv("XAI_API_KEY"):
            print("\n⚠️  No XAI_API_KEY found. Set it to run the demos:")
            print("   export XAI_API_KEY=your-key")
            print("\nTo start the server instead, run:")
            print("   python -m examples.sample_agent server")
            sys.exit(1)

        asyncio.run(demo_single_turn())
        asyncio.run(demo_streaming())

        print("\nTo start the server, run:")
        print("   python -m examples.sample_agent server")
