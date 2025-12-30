# Egile Agent Core

A turnkey AI agent framework with multi-provider LLM support and plugin architecture.

## Features

- **Multi-Provider LLM Support**: xAI (Grok), OpenAI, Azure OpenAI, and Mistral
- **Plugin Architecture**: Extensible through Python entry points
- **FastAPI Server**: Production-ready runtime for serving agents
- **Async-First**: Built for performant, non-blocking operations
- **Agent UI Integration**: Compatible with Agent UI for beautiful chat interfaces

## Installation

```bash
pip install egile-agent-core
```

For Mistral support:
```bash
pip install egile-agent-core[mistral]
```

## Quick Start

```python
from egile_agent_core import Agent
from egile_agent_core.models import XAI
from egile_agent_core.server import AgentServer

# Create an agent with xAI (Grok)
agent = Agent(
    name="my-agent",
    model=XAI(model="grok-4-1-fast-reasoning"),
    system_prompt="You are a helpful assistant.",
)

# Create and run the server
server = AgentServer(agents=[agent])
server.serve()
```

## Configuration

Set environment variables for your LLM provider:

```bash
# xAI (preferred)
export XAI_API_KEY=your-xai-key

# OpenAI
export OPENAI_API_KEY=your-openai-key

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=your-azure-key
export AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Mistral
export MISTRAL_API_KEY=your-mistral-key
```

Or create a `.env` file in your project root:
```bash
XAI_API_KEY=your-xai-key
```

## Running the Server

### Option 1: Using the Example Script (Quickest)

Run the provided example with pre-configured agents:

```bash
python examples/chatbot_with_agentui.py
```

This starts a server on port 8000 with multiple agents and provides Agent UI integration.

### Option 2: Creating Your Own Server

Create a Python script:

```python
from egile_agent_core import Agent
from egile_agent_core.models import XAI
from egile_agent_core.server import AgentServer

# Create an agent
agent = Agent(
    name="my-agent",
    model=XAI(model="grok-4-1-fast-reasoning"),
    system_prompt="You are a helpful assistant.",
)

# Create and run the server
server = AgentServer(agents=[agent])
server.serve(host="0.0.0.0", port=8000)
```

Then run:
```bash
python your_script.py
```

### Option 3: Using Uvicorn Directly

For more control (with auto-reload for development):

```bash
uvicorn your_module:app --host 0.0.0.0 --port 8000 --reload
```

### Server Configuration

The `serve()` method accepts these parameters:

- `host`: Host to bind to (default: `"0.0.0.0"`)
- `port`: Port to bind to (default: `8000`)
- `reload`: Enable auto-reload for development (default: `False`)
- `log_level`: Uvicorn log level (default: `"info"`)

### Accessing Your Server

Once running, you can access:
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Original API**: http://localhost:8000/v1/
- **Agent UI Compatible API**: http://localhost:8000/

## API Endpoints

Once the server is running:

### Original API (v1)
- `GET /v1/agents` - List available agents
- `GET /v1/agents/{agent_name}` - Get agent details
- `POST /v1/agents/{agent_name}/chat` - Chat with an agent
- `POST /v1/agents/{agent_name}/stream` - Stream chat responses (SSE)

### Agent UI Compatible API
- `GET /agents` - List agents (Agent UI format)
- `POST /agents/{agent_id}/runs` - Run agent with streaming
- `GET /health` - Health check
- `GET /sessions` - List sessions
- `DELETE /sessions/{session_id}` - Delete session

See [AGENT_UI_INTEGRATION.md](AGENT_UI_INTEGRATION.md) for connecting with Agent UI.

## Creating a Plugin

```python
from egile_agent_core.plugins import Plugin

class MyPlugin(Plugin):
    @property
    def name(self) -> str:
        return "my-plugin"
    
    async def on_message_received(self, message: str) -> str:
        # Pre-process messages
        return message
    
    async def on_response_generated(self, response: str) -> str:
        # Post-process responses
        return response
```

Register in your `pyproject.toml`:
```toml
[project.entry-points."egile_agent_core.plugins"]
my_plugin = "my_package:MyPlugin"
```

## License

MIT
