"""Example chatbot using Egile Agent Core with AgentOS and Agent UI integration.

This example demonstrates how to create a chatbot using Agno's AgentOS framework
with egile-agent-core's multi-provider LLM support.

The server provides Agent UI compatible endpoints and automatic session management.

To use this chatbot:
1. Set your API keys in the .env file
2. Run this script: python examples/chatbot_with_agentui.py
3. Navigate to the Agent UI and connect to http://localhost:8000
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

from egile_agent_core.models import OpenAI, XAI
from egile_agent_core.server import create_agent_os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("LOG_LEVEL") == "DEBUG" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Configure agents with egile models
agents_config = [
    {
        "name": "helpful-assistant",
        "model": XAI(model="grok-4-1-fast-reasoning") if os.getenv("XAI_API_KEY") else OpenAI(model="gpt-4o-mini"),
        "description": "A helpful AI assistant that can answer questions and help with tasks.",
        "instructions": [
            "You are a helpful, friendly AI assistant.",
            "Provide clear, concise answers and always be respectful.",
            "If you don't know something, admit it honestly.",
        ],
        "markdown": True,
    },
    {
        "name": "code-expert",
        "model": XAI(model="grok-4-1-fast-reasoning") if os.getenv("XAI_API_KEY") else OpenAI(model="gpt-4o-mini"),
        "description": "An expert programmer who can help with coding questions and debugging.",
        "instructions": [
            "You are an expert programmer with deep knowledge of multiple languages.",
            "Help users write better code, debug issues, and explain technical concepts clearly.",
            "Always provide code examples when relevant.",
        ],
        "markdown": True,
    },
    {
        "name": "creative-writer",
        "model": XAI(model="grok-4-1-fast-reasoning") if os.getenv("XAI_API_KEY") else OpenAI(model="gpt-4o-mini"),
        "description": "A creative writing assistant for stories, poems, and content creation.",
        "instructions": [
            "You are a creative writing assistant with a flair for storytelling.",
            "Help users craft engaging narratives, poetry, and creative content.",
            "Be imaginative, expressive, and encourage creativity.",
        ],
        "markdown": True,
    },
]

# Create AgentOS
agent_os = create_agent_os(
    agents_config=agents_config,
    os_id="chatbot-agent-os",
    description="AI-powered chatbots with multi-provider LLM support",
    db_file="chatbot_os.db",
)

# Get the FastAPI app
app = agent_os.get_app()

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Chatbot Server with AgentOS Starting")
    print("=" * 60)
    print()
    print(f"Available Agents: {', '.join(config['name'] for config in agents_config)}")
    print()
    print("API Endpoints:")
    print("  - Agent UI API: http://localhost:8000/")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print()
    print("Agent UI Integration:")
    print("  1. Navigate to your Agent UI directory")
    print("  2. Update .env.local with: NEXT_PUBLIC_AGENTOS_URL=http://localhost:8000")
    print("  3. Run: pnpm dev")
    print("  4. Open http://localhost:3000 in your browser")
    print()
    print("Database:")
    print("  - Sessions stored in: chatbot_os.db")
    print()
    print("=" * 60)
    print()

    # Start the server using uvicorn directly
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
