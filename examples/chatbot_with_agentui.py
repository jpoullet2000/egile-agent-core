"""Example chatbot using Egile Agent Core with Agent UI integration.

This example demonstrates how to create a chatbot that works with Agent UI.
The server provides both the original API endpoints and Agent UI compatible endpoints.

To use this chatbot:
1. Set your API keys in the .env file
2. Run this script: python examples/chatbot_with_agentui.py
3. Navigate to the Agent UI and connect to http://localhost:7860
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from egile_agent_core import Agent
from egile_agent_core.models import OpenAI, XAI
from egile_agent_core.server import AgentServer

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def main():
    """Create and serve chatbot agents."""
    # Create multiple agents with different personalities
    agents = [
        Agent(
            name="helpful-assistant",
            model=XAI(model="grok-4-1-fast-reasoning") if os.getenv("XAI_API_KEY") else OpenAI(model="gpt-4o-mini"),
            description="A helpful AI assistant that can answer questions and help with tasks.",
            system_prompt=(
                "You are a helpful, friendly AI assistant. "
                "Provide clear, concise answers and always be respectful. "
                "If you don't know something, admit it honestly."
            ),
        ),
        Agent(
            name="code-expert",
            model=XAI(model="grok-4-1-fast-reasoning") if os.getenv("XAI_API_KEY") else OpenAI(model="gpt-4o-mini"),
            description="An expert programmer who can help with coding questions and debugging.",
            system_prompt=(
                "You are an expert programmer with deep knowledge of multiple languages. "
                "Help users write better code, debug issues, and explain technical concepts clearly. "
                "Always provide code examples when relevant."
            ),
        ),
        Agent(
            name="creative-writer",
            model=XAI(model="grok-4-1-fast-reasoning") if os.getenv("XAI_API_KEY") else OpenAI(model="gpt-4o-mini"),
            description="A creative writing assistant for stories, poems, and content creation.",
            system_prompt=(
                "You are a creative writing assistant with a flair for storytelling. "
                "Help users craft engaging narratives, poetry, and creative content. "
                "Be imaginative, expressive, and encourage creativity."
            ),
        ),
    ]

    # Create the server
    server = AgentServer(
        agents=agents,
        title="Chatbot Server with Agent UI",
        description="AI-powered chatbots compatible with Agent UI",
        version="1.0.0",
        cors_origins=["*"],  # Allow all origins for development
    )

    print("=" * 60)
    print("🤖 Chatbot Server Starting")
    print("=" * 60)
    print()
    print(f"Available Agents: {', '.join(agent.name for agent in agents)}")
    print()
    print("API Endpoints:")
    print("  - Original API: http://localhost:8000/v1/")
    print("  - Agent UI API: http://localhost:8000/")
    print("  - API Docs: http://localhost:8000/docs")
    print()
    print("Agent UI Integration:")
    print("  1. Navigate to your Agent UI directory")
    print("  2. Update .env.local with: NEXT_PUBLIC_AGENTOS_URL=http://localhost:8000")
    print("  3. Run: pnpm dev")
    print("  4. Open http://localhost:3000 in your browser")
    print()
    print("=" * 60)
    print()

    # Start the server
    server.serve(host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
