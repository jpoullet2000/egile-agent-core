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
from egile_agent_core.models import XAI
from egile_agent_core.server import create_agent_os

# Configure an agent with xAI (Grok)
agents_config = [{
    "name": "my-agent",
    "model": XAI(model="grok-4-1-fast-reasoning"),
    "instructions": ["You are a helpful assistant."],
    "description": "A helpful AI assistant",
}]

# Create and run the AgentOS server
agent_os = create_agent_os(agents_config)
agent_os.serve()
```

### Alternative: Using AgentServer (Backward Compatibility)

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

### Option 1: Using create_agent_os() (Recommended)

The modern approach using Agno's AgentOS framework:

```python
from egile_agent_core.models import XAI
from egile_agent_core.server import create_agent_os

# Configure agents
agents_config = [
    {
        "name": "my-agent",
        "model": XAI(model="grok-4-1-fast-reasoning"),
        "instructions": ["You are a helpful assistant."],
        "description": "A helpful AI assistant",
        "markdown": True,
    }
]

# Create and run AgentOS
agent_os = create_agent_os(
    agents_config=agents_config,
    os_id="my-agent-os",
    description="My AI Agent OS",
    db_file="agent_os.db",  # Persistent session storage
    port=8000,
)

agent_os.serve(reload=True)  # reload=True for development
```

**Benefits of AgentOS:**
- Automatic Agent UI compatibility
- Built-in SQLite session persistence
- Session management and conversation history
- Can connect to Agno control plane for monitoring

### Option 2: Using the Example Script (Quickest)

Run the provided example with pre-configured agents:

```bash
python examples/chatbot_with_agentui.py
```

This starts an AgentOS server on port 8000 with multiple agents.

### Option 3: Using AgentServer (Backward Compatibility)

For backward compatibility with existing code:

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

### AgentOS Configuration

The `create_agent_os()` function accepts:

- `agents_config`: List of agent configuration dictionaries
- `os_id`: Unique ID for this AgentOS instance
- `description`: Description of the AgentOS
- `db_file`: Path to SQLite database for session storage
- `port`: Default port (can override in serve())

### Accessing Your Server

Once running, you can access:
- **API Documentation**: http://localhost:8000/docs
- **Agent UI Compatible API**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health
- **Configuration**: http://localhost:8000/config

## API Endpoints

AgentOS provides these endpoints automatically:

### Agent Management
- `GET /agents` - List all available agents
- `GET /agents/{agent_id}` - Get agent details

### Agent Interactions
- `POST /agents/{agent_id}/runs` - Start a conversation (streaming)
- `GET /agents/{agent_id}/runs/{run_id}` - Get run details

### Session Management
- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session details
- `GET /sessions/{session_id}/runs` - Get conversation history
- `DELETE /sessions/{session_id}` - Delete a session

### System
- `GET /health` - Health check
- `GET /config` - View AgentOS configuration
- `GET /docs` - Interactive API documentation

See [AGENT_UI_INTEGRATION.md](AGENT_UI_INTEGRATION.md) for Agent UI setup.

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
