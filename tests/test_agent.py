"""Tests for the multimodal agent."""

import pytest
from langchain_core.messages import HumanMessage

from src.agent import create_agent
from src.models import create_multimodal_message


@pytest.mark.asyncio
async def test_agent_creation():
    """Test agent can be created."""
    agent = await create_agent(use_mcp=False)
    assert agent is not None
    assert agent.model is not None


@pytest.mark.asyncio
async def test_text_message():
    """Test agent can process text messages."""
    agent = await create_agent(use_mcp=False)

    message = HumanMessage(content="Hello, how are you?")
    response = await agent.ainvoke(message)

    assert response is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_multimodal_message_creation():
    """Test multimodal message creation."""
    message = create_multimodal_message(
        text="What is this?",
        images=None,
        audio=None,
        video=None,
    )

    assert message is not None
    assert isinstance(message, HumanMessage)


def test_agent_state():
    """Test agent state structure."""
    from src.agent import AgentState

    state = AgentState(messages=[], user_id="test_user")
    assert state["user_id"] == "test_user"
    assert len(state["messages"]) == 0
