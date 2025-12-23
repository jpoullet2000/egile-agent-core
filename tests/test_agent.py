"""Tests for core Agent class."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from egile_agent_core.agent import Agent, Message, AgentResponse
from egile_agent_core.models.base import BaseLLM, LLMResponse
from egile_agent_core.plugins.base import Plugin


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self, model: str = "mock-model", **kwargs):
        super().__init__(model=model, **kwargs)

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(
            content="Mock response",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def stream(self, messages: list[dict[str, str]]):
        for word in ["Mock", " ", "streaming", " ", "response"]:
            yield word


class MockPlugin(Plugin):
    """Mock plugin for testing."""

    def __init__(self):
        self.hook_calls = []

    @property
    def name(self) -> str:
        return "mock-plugin"

    async def on_agent_start(self, agent):
        self.hook_calls.append("on_agent_start")

    async def on_message_received(self, message: str, **kwargs) -> str:
        self.hook_calls.append(("on_message_received", message))
        return f"[preprocessed] {message}"

    async def on_response_generated(self, response: str, **kwargs) -> str:
        self.hook_calls.append(("on_response_generated", response))
        return f"[postprocessed] {response}"


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        msg = Message(role="assistant", content="Hi there")
        assert msg.to_dict() == {"role": "assistant", "content": "Hi there"}


class TestAgentResponse:
    """Tests for AgentResponse class."""

    def test_response_creation(self):
        response = AgentResponse(
            content="Test content",
            agent_name="test-agent",
            model="test-model",
            usage={"total_tokens": 100},
        )
        assert response.content == "Test content"
        assert response.agent_name == "test-agent"
        assert response.model == "test-model"
        assert response.usage == {"total_tokens": 100}


class TestAgent:
    """Tests for Agent class."""

    def test_agent_creation(self):
        llm = MockLLM()
        agent = Agent(
            name="test-agent",
            model=llm,
            description="Test description",
            system_prompt="You are a test assistant.",
        )

        assert agent.name == "test-agent"
        assert agent.description == "Test description"
        assert agent.system_prompt == "You are a test assistant."
        assert len(agent.history) == 1  # System message
        assert agent.history[0].role == "system"

    def test_agent_without_system_prompt(self):
        llm = MockLLM()
        agent = Agent(name="test-agent", model=llm)

        assert agent.system_prompt == ""
        assert len(agent.history) == 0

    @pytest.mark.asyncio
    async def test_agent_run(self):
        llm = MockLLM()
        agent = Agent(name="test-agent", model=llm)

        response = await agent.run("Hello")

        assert response.content == "Mock response"
        assert response.agent_name == "test-agent"
        assert response.model == "mock/mock-model"
        assert len(agent.history) == 2  # User + Assistant

    @pytest.mark.asyncio
    async def test_agent_run_with_plugins(self):
        llm = MockLLM()
        plugin = MockPlugin()
        agent = Agent(name="test-agent", model=llm, plugins=[plugin])

        response = await agent.run("Hello")

        # Check plugin hooks were called
        assert "on_agent_start" in plugin.hook_calls
        assert ("on_message_received", "Hello") in plugin.hook_calls
        assert ("on_response_generated", "Mock response") in plugin.hook_calls

        # Check preprocessing was applied
        assert agent.history[0].content == "[preprocessed] Hello"

        # Check postprocessing was applied
        assert "[postprocessed]" in response.content

    @pytest.mark.asyncio
    async def test_agent_stream(self):
        llm = MockLLM()
        agent = Agent(name="test-agent", model=llm)

        chunks = []
        async for chunk in agent.stream("Hello"):
            chunks.append(chunk)

        assert chunks == ["Mock", " ", "streaming", " ", "response"]
        assert len(agent.history) == 2

    def test_agent_clear_history(self):
        llm = MockLLM()
        agent = Agent(
            name="test-agent",
            model=llm,
            system_prompt="System",
        )
        agent.history.append(Message(role="user", content="Hello"))
        agent.history.append(Message(role="assistant", content="Hi"))

        agent.clear_history()

        # System message should be preserved
        assert len(agent.history) == 1
        assert agent.history[0].role == "system"

    def test_agent_get_info(self):
        llm = MockLLM()
        plugin = MockPlugin()
        agent = Agent(
            name="test-agent",
            model=llm,
            description="Test",
            plugins=[plugin],
        )

        info = agent.get_info()

        assert info["name"] == "test-agent"
        assert info["description"] == "Test"
        assert info["model"] == "mock/mock-model"
        assert info["provider"] == "mock"
        assert "mock-plugin" in info["plugins"]
