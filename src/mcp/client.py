"""MCP client for integrating Model Context Protocol servers with LangGraph."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from loguru import logger

from ..config import settings


class MCPClientManager:
    """
    Manages connections to multiple MCP servers and provides tools to LangGraph agents.

    Features:
    - Multi-server support (stdio, HTTP, SSE transports)
    - Automatic tool discovery and loading
    - Connection pooling and lifecycle management
    - Error handling and reconnection
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP client manager.

        Args:
            config_path: Path to MCP servers configuration JSON file
        """
        self.config_path = config_path or settings.mcp_servers_config
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MCP client and connect to all configured servers."""
        if self._initialized:
            logger.warning("MCP client already initialized")
            return

        try:
            # Load server configuration
            config = self._load_config()
            servers_config = config.get("servers", {})

            if not servers_config:
                logger.warning("No MCP servers configured")
                self._initialized = True
                return

            # Create multi-server client
            self.client = MultiServerMCPClient(servers_config)

            # Load tools from all servers
            logger.info(f"Loading tools from {len(servers_config)} MCP servers...")
            self.tools = await self.client.get_tools()

            logger.info(
                f"✓ MCP client initialized with {len(self.tools)} tools from "
                f"{len(servers_config)} servers"
            )
            self._log_available_tools()

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise

    def _load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration from JSON file."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            logger.warning(f"MCP config file not found: {config_path}")
            return {"servers": {}}

        with open(config_path, "r") as f:
            config = json.load(f)

        return config

    def _log_available_tools(self) -> None:
        """Log information about available MCP tools."""
        if not self.tools:
            return

        logger.info("Available MCP tools:")
        for tool in self.tools:
            logger.info(f"  - {tool.name}: {tool.description}")

    async def get_tools(self) -> List[BaseTool]:
        """
        Get all available MCP tools.

        Returns:
            List of LangChain-compatible tools
        """
        if not self._initialized:
            await self.initialize()

        return self.tools

    async def get_tools_by_server(self, server_name: str) -> List[BaseTool]:
        """
        Get tools from a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            List of tools from the specified server
        """
        if not self._initialized:
            await self.initialize()

        if not self.client:
            return []

        try:
            async with self.client.session(server_name) as session:
                tools = await load_mcp_tools(session)
                return tools
        except Exception as e:
            logger.error(f"Failed to get tools from server {server_name}: {e}")
            return []

    async def close(self) -> None:
        """Close all MCP server connections."""
        if self.client:
            try:
                await self.client.close()
                logger.info("MCP client connections closed")
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")

        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global MCP client instance
_mcp_client: Optional[MCPClientManager] = None


async def get_mcp_client() -> MCPClientManager:
    """
    Get or create the global MCP client instance.

    Returns:
        MCPClientManager instance
    """
    global _mcp_client

    if _mcp_client is None:
        _mcp_client = MCPClientManager()
        await _mcp_client.initialize()

    return _mcp_client


async def get_all_mcp_tools() -> List[BaseTool]:
    """
    Convenience function to get all MCP tools.

    Returns:
        List of all available MCP tools
    """
    client = await get_mcp_client()
    return await client.get_tools()
