# Agent UI Integration Guide

This guide explains how to connect your egile-agent-core chatbot backend to the Agent UI frontend.

## Prerequisites

1. **egile-agent-core backend** - This project (already set up)
2. **Agent UI frontend** - Located at `C:\Users\jeanb\OneDrive\Documents\projects\agent-ui`
3. **API Keys** - Set in your `.env` file (XAI_API_KEY, OPENAI_API_KEY, etc.)

## Quick Start

### Step 1: Start the Backend

In this project directory (`egile-agent-core`):

```bash
# Activate virtual environment
.\agent-core-env\Scripts\Activate.ps1

# Run the chatbot server
python examples\chatbot_with_agentui.py
```

The server will start on `http://localhost:8000` with:
- Agent UI compatible endpoints (AgentOS)
- Automatic session management with SQLite
- Interactive docs at `/docs`

### Step 2: Configure Agent UI

Navigate to your Agent UI directory:

```bash
cd C:\Users\jeanb\OneDrive\Documents\projects\agent-ui
```

Create or update `.env.local`:

```env
# Point Agent UI to your egile-agent-core backend
NEXT_PUBLIC_AGENTOS_URL=http://localhost:8000

# Optional: Set auth token if needed
# NEXT_PUBLIC_AUTH_TOKEN=your-token-here
```

### Step 3: Start Agent UI

```bash
# Install dependencies (first time only)
pnpm install

# Start the development server
pnpm dev
```

Agent UI will start on `http://localhost:3000`

### Step 4: Use the Chatbot

1. Open your browser to `http://localhost:3000`
2. The UI will automatically discover available agents from your backend
3. Select an agent (helpful-assistant, code-expert, or creative-writer)
4. Start chatting!

## Available Agents

The example creates three pre-configured agents:

1. **helpful-assistant** - A general-purpose helpful AI assistant
2. **code-expert** - Specializes in programming and debugging
3. **creative-writer** - Focused on creative writing and storytelling

## API Endpoints

Your backend exposes the following Agent UI compatible endpoints:

- `GET /agents` - List all available agents
- `POST /agents/{agent_id}/runs` - Start a conversation (streaming)
- `GET /health` - Health check
- `GET /sessions` - List sessions
- `GET /sessions/{session_id}/runs` - Get conversation history
- `DELETE /sessions/{session_id}` - Delete a session

Original API endpoints are still available at `/v1/*`

## Customization

### Adding More Agents

Edit `examples/chatbot_with_agentui.py` to add more agents:

```python
{
    "name": "your-agent-name",
    "model": XAI(model="grok-4-1-fast-reasoning"),
    "description": "A brief description shown in Agent UI",
    "instructions": [
        "Your agent's first instruction...",
        "Additional instructions...",
    ],
    "markdown": True,  # Enable markdown formatting
}
```

### Changing Models

You can use different models by switching the model provider:

```python
from egile_agent_core.models import OpenAI, XAI, AzureOpenAI, Mistral

# OpenAI
model=OpenAI(model="gpt-4o")

# XAI
model=XAI(model="grok-4-1-fast-reasoning")

# Azure OpenAI
model=AzureOpenAI(
    deployment_name="your-deployment",
    endpoint="https://your-resource.openai.azure.com/",
)
```

### Using create_agent_os() Directly

For more control, create your own script:

```python
from egile_agent_core.models import XAI
from egile_agent_core.server import create_agent_os

agents_config = [
    {
        "name": "my-agent",
        "model": XAI(model="grok-4-1-fast-reasoning"),
        "description": "My custom agent",
        "instructions": ["Custom instructions..."],
        "markdown": True,
        "debug_mode": False,  # Set to True for debugging
    }
]

agent_os = create_agent_os(
    agents_config=agents_config,
    os_id="my-custom-os",
    description="My Custom AgentOS",
    db_file="my_custom_os.db",
    port=8000,
)

agent_os.serve(reload=True)  # Auto-reload for development
```

## Troubleshooting

### Agent UI can't connect to backend

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check `.env.local` has correct URL
3. Ensure CORS is enabled (it is by default)

### No agents showing in UI

1. Check backend logs for errors
2. Verify `/agents` endpoint: `curl http://localhost:8000/agents`
3. Make sure you have valid API keys in `.env`

### Streaming not working

1. Ensure you're using SSE-compatible browser
2. Check browser console for errors
3. Verify the `/agents/{id}/runs` endpoint is accessible

## Production Deployment

For production use:

1. **Use environment variables** for configuration
2. **Set a production database** path
3. **Disable auto-reload**
4. **Use a production WSGI server** (Gunicorn, etc.)
5. **Enable HTTPS** for secure communication
6. **Connect to Agno Control Plane** for monitoring (optional)

Example production setup:

```python
from egile_agent_core.models import XAI
from egile_agent_core.server import create_agent_os
import os

agents_config = [
    {
        "name": "production-agent",
        "model": XAI(model="grok-4-1-fast-reasoning"),
        "instructions": ["You are a helpful assistant."],
        "description": "Production agent",
        "markdown": True,
        "debug_mode": False,
    }
]

agent_os = create_agent_os(
    agents_config=agents_config,
    os_id="production-os",
    description="Production AgentOS",
    db_file="/var/lib/agent_os/production.db",
    port=int(os.getenv("PORT", "8000")),
)

# Get the FastAPI app for use with Gunicorn
app = agent_os.get_app()

# Or run directly
if __name__ == "__main__":
    agent_os.serve(reload=False)
```

Run with Gunicorn:
```bash
gunicorn your_module:app --workers 4 --bind 0.0.0.0:8000
```

## Learn More

- [Egile Agent Core Documentation](../README.md)
- [Agent UI GitHub](https://github.com/agno-agi/agent-ui)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
