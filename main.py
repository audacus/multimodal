#!/usr/bin/env python3
"""Main entry point for the multimodal AI agent Telegram bot."""

import asyncio
import sys

from loguru import logger

from src.bot import run_bot
from src.config import settings


def setup_logging():
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # Add file logger
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="7 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


async def main():
    """Main application entry point."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("🤖 Multimodal AI Agent Starting")
    logger.info("=" * 60)
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"API Base: {settings.model_api_base}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"MCP Config: {settings.mcp_servers_config}")
    logger.info("=" * 60)

    try:
        # Run the Telegram bot
        await run_bot()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
