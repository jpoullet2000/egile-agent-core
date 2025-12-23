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
    model=XAI(model="grok-beta"),
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
