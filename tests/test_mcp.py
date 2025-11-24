"""Tests for MCP client."""

import pytest

from src.mcp import MCPClientManager


@pytest.mark.asyncio
async def test_mcp_client_creation():
    """Test MCP client can be created."""
    client = MCPClientManager()
    assert client is not None


@pytest.mark.asyncio
async def test_mcp_client_initialization():
    """Test MCP client initialization."""
    async with MCPClientManager() as client:
        # Should initialize without errors even with no servers
        assert client._initialized


@pytest.mark.asyncio
async def test_get_tools_empty():
    """Test getting tools with no servers configured."""
    async with MCPClientManager() as client:
        tools = await client.get_tools()
        # Should return empty list if no servers configured
        assert isinstance(tools, list)
