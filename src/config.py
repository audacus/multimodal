"""Configuration management for the multimodal AI agent."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram Bot
    telegram_bot_token: str = Field(..., description="Telegram bot token")

    # Model Configuration
    model_name: str = Field(default="Qwen/Qwen3-Omni", description="Model name")
    model_api_base: str = Field(
        default="http://localhost:8000/v1", description="Model API base URL"
    )
    model_api_key: Optional[str] = Field(default=None, description="API key if required")

    # LLM Settings
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")

    # MCP Configuration
    mcp_servers_config: str = Field(
        default="config/mcp_servers.json", description="Path to MCP servers config"
    )

    # Application Settings
    log_level: str = Field(default="INFO", description="Logging level")
    max_conversation_history: int = Field(
        default=10, description="Max messages to keep in conversation history"
    )
    enable_memory: bool = Field(default=True, description="Enable conversation memory")

    # GPU Settings
    cuda_visible_devices: str = Field(default="0", description="CUDA visible devices")
    torch_dtype: str = Field(default="float16", description="Torch dtype for model")

    def load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP servers configuration from JSON file."""
        config_path = Path(self.mcp_servers_config)
        if not config_path.exists():
            return {"servers": {}}

        with open(config_path, "r") as f:
            return json.load(f)


# Global settings instance
settings = Settings()
