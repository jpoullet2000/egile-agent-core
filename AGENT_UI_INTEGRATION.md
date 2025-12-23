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

The server will start on `http://localhost:7860` with:
- Agent UI compatible endpoints at the root (`/agents`, `/health`, etc.)
- Original API endpoints at `/v1/agents`
- Interactive docs at `/docs`

### Step 2: Configure Agent UI

Navigate to your Agent UI directory:

```bash
cd C:\Users\jeanb\OneDrive\Documents\projects\agent-ui
```

Create or update `.env.local`:

```env
# Point Agent UI to your egile-agent-core backend
NEXT_PUBLIC_AGENTOS_URL=http://localhost:7860

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
Agent(
    name="your-agent-name",
    model=XAI(model="grok-beta"),
    description="A brief description shown in Agent UI",
    system_prompt="Your agent's system prompt...",
)
```

### Changing Models

You can use different models by switching the model provider:

```python
from egile_agent_core.models import OpenAI, XAI, AzureOpenAI, Mistral

# OpenAI
model=OpenAI(model="gpt-4o")

# XAI
model=XAI(model="grok-beta")

# Azure OpenAI
model=AzureOpenAI(
    deployment_name="your-deployment",
    endpoint="https://your-resource.openai.azure.com/",
)
```

### Adding Plugins

Enhance your agents with custom plugins:

```python
from egile_agent_core.plugins.base import Plugin

class MyPlugin(Plugin):
    name = "my-plugin"
    
    async def before_request(self, message: str, **kwargs) -> str:
        # Modify message before sending to LLM
        return message

agent = Agent(
    name="enhanced-agent",
    model=XAI(model="grok-beta"),
    plugins=[MyPlugin()],
)
```

## Troubleshooting

### Agent UI can't connect to backend

1. Verify backend is running: `curl http://localhost:7860/health`
2. Check `.env.local` has correct URL
3. Ensure CORS is enabled (it is by default)

### No agents showing in UI

1. Check backend logs for errors
2. Verify `/agents` endpoint: `curl http://localhost:7860/agents`
3. Make sure you have valid API keys in `.env`

### Streaming not working

1. Ensure you're using SSE-compatible browser
2. Check browser console for errors
3. Verify the `/agents/{id}/runs` endpoint is accessible

## Production Deployment

For production use:

1. **Use environment variables** for configuration
2. **Add authentication** to protect your API
3. **Use a proper database** for session storage (replace in-memory dict)
4. **Set specific CORS origins** instead of allowing all
5. **Enable HTTPS** for secure communication

Example production server setup:

```python
server = AgentServer(
    agents=agents,
    cors_origins=["https://your-domain.com"],
)

server.serve(
    host="0.0.0.0",
    port=8000,
    reload=False,
    log_level="warning",
)
```

## Learn More

- [Egile Agent Core Documentation](../README.md)
- [Agent UI GitHub](https://github.com/agno-agi/agent-ui)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
