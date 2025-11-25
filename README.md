# Multimodal AI Agent with LangGraph & Telegram

A bi-directional multimodal AI agent that processes text, images, audio, and video through a Telegram bot interface. Built with LangChain/LangGraph v1.x, powered by Qwen3 Omni, and deployable on DigitalOcean GPU Droplets.

## Features

### 🎯 Core Capabilities
- **Multimodal Processing**: Text, images, audio, and video understanding
- **Tool Usage**: Integration with MCP (Model Context Protocol) servers
- **Bi-directional Communication**: Real-time conversations via Telegram
- **State Management**: LangGraph-powered conversation flow
- **GPU Optimized**: Efficient inference on DigitalOcean GPU Droplets

### 🛠️ Technology Stack
- **Backend**: LangChain/LangGraph v1.x
- **Frontend**: Telegram Bot (python-telegram-bot)
- **Model**: Qwen3 Omni (multimodal LLM)
- **Tools**: MCP servers for external integrations
- **Deployment**: Docker + NVIDIA GPU support

## Architecture

```
┌─────────────────┐
│  Telegram Bot   │
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangGraph      │
│  Agent          │
│  ┌───────────┐  │
│  │  Agent    │  │
│  │  Node     │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │  Tools    │  │
│  │  Node     │  │
│  └───────────┘  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────┐   ┌─────┐
│Qwen3│   │ MCP │
│Omni │   │Srvs │
└─────┘   └─────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.10-3.13 (recommended: 3.13)
- NVIDIA GPU (for local deployment)
- Docker & Docker Compose
- Telegram Bot Token ([Create one with @BotFather](https://t.me/BotFather))

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd multimodal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Required environment variables:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
MODEL_NAME=Qwen/Qwen3-Omni
MODEL_API_BASE=http://localhost:8000/v1  # Or your GPU droplet IP
```

### 4. Run Locally

```bash
# Start the bot
python main.py
```

## Deployment on DigitalOcean GPU Droplet

### Option 1: Automated Setup

```bash
# Run setup script
chmod +x deploy/digitalocean_setup.sh
./deploy/digitalocean_setup.sh

# Configure environment
nano .env

# Start with Docker Compose
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### Option 2: Systemd Service

```bash
# Create systemd service
chmod +x deploy/systemd_service.sh
./deploy/systemd_service.sh

# Enable and start service
sudo systemctl enable multimodal-agent
sudo systemctl start multimodal-agent

# Check status
sudo systemctl status multimodal-agent
```

## MCP Server Configuration

Configure MCP servers in `config/mcp_servers.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "transport": "stdio",
      "description": "File system operations"
    },
    "custom_api": {
      "transport": "streamable_http",
      "url": "http://localhost:8001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      },
      "description": "Custom API integration"
    }
  }
}
```

### Available Transport Types
- **stdio**: Local process communication
- **streamable_http**: HTTP-based (for remote servers)
- **sse**: Server-Sent Events

## Usage Examples

### Text Conversation
```
User: Explain quantum computing in simple terms
Bot: [Provides explanation using Qwen3 Omni]
```

### Image Analysis
```
User: [Sends photo] What's in this image?
Bot: [Analyzes and describes the image content]
```

### Audio Processing
```
User: [Sends audio file] Transcribe this
Bot: [Provides transcription using multimodal capabilities]
```

### Video Understanding
```
User: [Sends video] Summarize this video
Bot: [Analyzes and summarizes video content]
```

### Tool Usage
```
User: Search for recent papers on AI
Bot: [Uses web search MCP tool to find and summarize papers]
```

## Project Structure

```
multimodal/
├── src/
│   ├── agent/          # LangGraph agent implementation
│   │   ├── graph.py    # Agent graph and state management
│   │   └── __init__.py
│   ├── bot/            # Telegram bot frontend
│   │   ├── telegram_bot.py
│   │   └── __init__.py
│   ├── mcp/            # MCP client integration
│   │   ├── client.py
│   │   └── __init__.py
│   ├── models/         # Qwen3 Omni model wrapper
│   │   ├── qwen_omni.py
│   │   └── __init__.py
│   ├── tools/          # Custom tools (optional)
│   ├── utils/          # Utility functions
│   └── config.py       # Configuration management
├── config/
│   └── mcp_servers.json # MCP server configuration
├── deploy/
│   ├── digitalocean_setup.sh
│   └── systemd_service.sh
├── tests/              # Unit tests
├── logs/               # Application logs
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose configuration
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Telegram Bot Commands

- `/start` - Start conversation and see welcome message
- `/help` - Show help and usage information
- `/clear` - Clear conversation history

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

### Adding Custom Tools

Create a new tool in `src/tools/`:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Description of what this tool does."""
    # Implementation
    return result
```

Then add it to the agent:

```python
from src.tools.my_tool import my_custom_tool

agent = await create_agent(
    custom_tools=[my_custom_tool],
    use_mcp=True
)
```

## Monitoring & Logging

Logs are stored in the `logs/` directory:
- `app.log` - General application logs
- `telegram_bot.log` - Telegram bot specific logs

View live logs:
```bash
tail -f logs/app.log
```

With Docker:
```bash
docker-compose logs -f
```

## Performance Optimization

### GPU Memory Management
Adjust in `.env`:
```bash
TORCH_DTYPE=float16  # or bfloat16 for better quality
CUDA_VISIBLE_DEVICES=0
```

### Conversation History
Limit memory usage:
```bash
MAX_CONVERSATION_HISTORY=10  # Keep last 10 messages
```

### Model Caching
Models are cached in Docker volumes to avoid re-downloading.

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA drivers
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Telegram Bot Not Responding
1. Check bot token in `.env`
2. Verify bot is running: `docker-compose ps`
3. Check logs: `docker-compose logs -f`

### MCP Tools Not Loading
1. Verify `config/mcp_servers.json` syntax
2. Check MCP server endpoints are accessible
3. Review logs for connection errors

### Out of Memory
- Reduce `MAX_TOKENS` in `.env`
- Use `TORCH_DTYPE=float16`
- Limit `MAX_CONVERSATION_HISTORY`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [Qwen Models](https://github.com/QwenLM/Qwen)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Resources

### Documentation
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Build multimodal agents using Gemini, Langchain, and LangGraph](https://cloud.google.com/blog/products/ai-machine-learning/build-multimodal-agents-using-gemini-langchain-and-langgraph)
- [LangGraph MCP Integration Guide](https://latenode.com/blog/langgraph-mcp-integration-complete-model-context-protocol-setup-guide-working-examples-2025)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Support

For issues and questions:
- GitHub Issues: [Your Repo Issues]
- Email: [Your Email]

---

**Note**: This project uses Qwen3 Omni. Ensure you comply with the model's license and usage terms.
