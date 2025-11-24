# Development Documentation

**Project**: Multimodal AI Agent with LangGraph & Telegram Bot
**Created**: 2025-11-22
**Status**: Production Ready
**Version**: 0.1.0

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Decisions](#architecture-decisions)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Implementation Details](#implementation-details)
6. [File Structure](#file-structure)
7. [Configuration](#configuration)
8. [Design Patterns](#design-patterns)
9. [Key Decisions](#key-decisions)
10. [Integration Points](#integration-points)
11. [Deployment Strategy](#deployment-strategy)
12. [Testing Strategy](#testing-strategy)
13. [Future Considerations](#future-considerations)

---

## Project Overview

### Goal
Create a bi-directional multimodal AI agent that can:
- Process text, images, audio, and video inputs
- Use external tools via Model Context Protocol (MCP) servers
- Communicate with users through Telegram
- Run efficiently on DigitalOcean GPU Droplets

### Requirements
- **Backend**: LangChain/LangGraph v1.x
- **Frontend**: Telegram bot
- **Model**: Qwen3 Omni (multimodal LLM)
- **Tools**: MCP server integration
- **Deployment**: Docker + GPU support

---

## Architecture Decisions

### 1. Framework Selection: LangGraph v1.x

**Decision**: Use LangGraph StateGraph for agent orchestration

**Rationale**:
- **State Management**: Built-in state handling for conversation context
- **Conditional Routing**: Automatic routing between agent and tool nodes
- **Tool Integration**: Native support for LangChain tools
- **Scalability**: Graph-based architecture allows easy extension
- **Modern Patterns**: Follows 2025 best practices for multi-agent systems

**Reference**: Based on [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)

### 2. Model Choice: Qwen3 Omni

**Decision**: Use Qwen3 Omni as the primary LLM

**Rationale**:
- **Multimodal**: Native support for text, image, audio, video
- **Open Source**: Can be self-hosted on GPU droplets
- **Performance**: Efficient inference with float16/bfloat16
- **API Compatible**: Supports both local and API-based deployment

**Implementation**: Custom LangChain wrapper (src/models/qwen_omni.py)

### 3. MCP for Tool Integration

**Decision**: Use Model Context Protocol (MCP) for tool calling

**Rationale**:
- **Standardized Protocol**: JSON-RPC 2.0 based, well-defined
- **Multi-Server Support**: Can integrate multiple tool sources
- **Transport Flexibility**: stdio, HTTP, SSE support
- **LangChain Integration**: Official adapters via langchain-mcp-adapters
- **Ecosystem**: Access to 250+ existing MCP servers

**Reference**: [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

### 4. Telegram as Frontend

**Decision**: Use Telegram Bot API (python-telegram-bot v21+)

**Rationale**:
- **Multimodal Support**: Native handling of text, photos, audio, video, documents
- **Bi-directional**: Real-time two-way communication
- **User-Friendly**: Familiar interface for end users
- **Rich Features**: Commands, inline keyboards, file handling
- **Async Support**: Compatible with async Python architecture

---

## Technology Stack

### Core Dependencies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Framework | LangChain | >=0.3.0 | LLM orchestration |
| Framework | LangGraph | >=0.2.0 | Agent workflow |
| Tools | langchain-mcp-adapters | >=0.1.0 | MCP integration |
| Bot | python-telegram-bot | >=21.0 | Telegram interface |
| Model | transformers | >=4.45.0 | Model loading |
| Compute | torch | >=2.0.0 | GPU inference |
| Inference | vllm | >=0.6.0 | Efficient serving |
| Media | PIL, opencv, pydub, moviepy | Latest | Multimodal processing |
| Config | pydantic-settings | >=2.0.0 | Configuration management |
| Logging | loguru | >=0.7.0 | Structured logging |

### Infrastructure

- **Containerization**: Docker with NVIDIA GPU support
- **Orchestration**: Docker Compose
- **Deployment**: DigitalOcean GPU Droplets
- **Service Management**: systemd (optional)

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Telegram Bot Layer                      │
│  (src/bot/telegram_bot.py)                                  │
│  - Message handling (text, image, audio, video)             │
│  - Media download & conversion                              │
│  - User session management                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Layer                     │
│  (src/agent/graph.py)                                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  StateGraph Workflow                                │   │
│  │                                                     │   │
│  │  ┌──────────┐   tools_condition   ┌──────────┐    │   │
│  │  │  Agent   ├───────────────────→ │  Tools   │    │   │
│  │  │  Node    │                      │  Node    │    │   │
│  │  │          │←─────────────────────┤          │    │   │
│  │  └────┬─────┘                      └──────────┘    │   │
│  │       │                                            │   │
│  │       │ no tools                                   │   │
│  │       ▼                                            │   │
│  │     [END]                                          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│   Model Layer        │  │   MCP Layer          │
│ (src/models/)        │  │ (src/mcp/)           │
│                      │  │                      │
│ Qwen3OmniModel       │  │ MultiServerMCPClient │
│ - Local inference    │  │ - stdio transport    │
│ - API inference      │  │ - HTTP transport     │
│ - Multimodal encode  │  │ - SSE transport      │
│ - GPU optimization   │  │ - Tool discovery     │
└──────────────────────┘  └──────────────────────┘
```

### Data Flow

1. **User Input** → Telegram receives message (text/media)
2. **Media Processing** → Download and convert to appropriate format
3. **Message Creation** → Create multimodal LangChain message
4. **Agent Invocation** → LangGraph processes message through workflow
5. **Agent Node** → Model generates response or tool calls
6. **Tool Execution** (if needed) → MCP tools execute
7. **Response Generation** → Final response created
8. **User Output** → Telegram sends response to user

---

## Implementation Details

### 1. LangGraph Agent (src/agent/graph.py)

**AgentState TypedDict**:
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
```

**Graph Structure**:
- **Entry Point**: `agent` node
- **Agent Node**: Calls model with tools bound
- **Conditional Edge**: `tools_condition` checks for tool calls
  - If tool calls → go to `tools` node
  - If no tool calls → END
- **Tools Node**: Executes MCP tools via `ToolNode`
- **Loop Back**: After tools execute, return to agent node

**Key Features**:
- System message prepending
- Conversation history limiting (configurable)
- Async and sync invocation methods
- Streaming support

### 2. Qwen3 Omni Model Wrapper (src/models/qwen_omni.py)

**Design**: Custom `BaseChatModel` implementation

**Dual Inference Modes**:
1. **Local Inference**:
   - Loads model via `transformers.AutoModelForCausalLM`
   - Uses `AutoProcessor` for multimodal inputs
   - GPU optimization with configurable dtype

2. **API Inference**:
   - Fallback to HTTP API endpoint
   - OpenAI-compatible chat completions format
   - Useful for remote GPU droplets

**Multimodal Message Encoding**:
- Text: Direct string
- Images: Base64-encoded in `image_url` format
- Audio: Base64-encoded in `audio_url` format
- Video: Base64-encoded in `video_url` format

**Helper Function**: `create_multimodal_message()`
- Accepts text + optional media files
- Handles PIL Images, file paths
- Returns properly formatted `HumanMessage`

### 3. MCP Client Manager (src/mcp/client.py)

**Architecture**: Singleton pattern with async context manager

**MultiServerMCPClient Integration**:
```python
client = MultiServerMCPClient({
    "server_name": {
        "command": "...",
        "args": [...],
        "transport": "stdio"
    }
})
tools = await client.get_tools()
```

**Transport Support**:
- **stdio**: Local process communication (for npx, python servers)
- **streamable_http**: HTTP-based (for remote servers)
- **sse**: Server-Sent Events (with custom headers)

**Lifecycle Management**:
- Lazy initialization
- Connection pooling
- Graceful shutdown
- Error handling and logging

### 4. Telegram Bot (src/bot/telegram_bot.py)

**Handler Structure**:
- **Commands**: `/start`, `/help`, `/clear`
- **Messages**: All media types via unified handler

**Media Processing**:
1. **Photos**: Downloaded to memory, converted to PIL Image
2. **Audio/Voice**: Downloaded to temp file, path passed to model
3. **Video**: Downloaded to temp file, path passed to model
4. **Documents**: MIME type detection, appropriate handling

**Session Management**:
- Temp files tracked per user ID
- Automatic cleanup on `/clear`
- File cleanup on shutdown

**Error Handling**:
- Try-catch around message processing
- User-friendly error messages
- Full error logging for debugging

### 5. Configuration (src/config.py)

**Pydantic Settings**:
- Loads from `.env` file
- Type validation
- Default values provided
- Environment variable mapping

**Key Settings**:
- `telegram_bot_token`: Bot authentication
- `model_name`: Qwen3 Omni model identifier
- `model_api_base`: API endpoint or local server
- `mcp_servers_config`: Path to MCP config JSON
- `max_conversation_history`: Memory limit
- `cuda_visible_devices`: GPU selection
- `torch_dtype`: float16/bfloat16

---

## File Structure

```
multimodal/
├── src/                          # Source code
│   ├── agent/                    # LangGraph agent implementation
│   │   ├── __init__.py           # Exports: MultimodalAgent, create_agent
│   │   └── graph.py              # Agent graph, state, node definitions
│   ├── bot/                      # Telegram bot frontend
│   │   ├── __init__.py           # Exports: TelegramBot, run_bot
│   │   └── telegram_bot.py       # Bot handlers, media processing
│   ├── mcp/                      # MCP client integration
│   │   ├── __init__.py           # Exports: MCPClientManager, get_tools
│   │   └── client.py             # MCP client, multi-server support
│   ├── models/                   # Model wrappers
│   │   ├── __init__.py           # Exports: Qwen3OmniModel, helpers
│   │   └── qwen_omni.py          # Qwen3 Omni LangChain wrapper
│   ├── tools/                    # Custom tools (empty, for extension)
│   ├── utils/                    # Utility functions (empty, for extension)
│   ├── __init__.py               # Package exports
│   └── config.py                 # Configuration management
├── config/                       # Configuration files
│   └── mcp_servers.json          # MCP server definitions
├── deploy/                       # Deployment scripts
│   ├── digitalocean_setup.sh     # Automated DigitalOcean setup
│   └── systemd_service.sh        # Systemd service installer
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_agent.py             # Agent tests
│   └── test_mcp.py               # MCP client tests
├── logs/                         # Log files (created at runtime)
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore patterns
├── README.md                     # User documentation
└── DEVELOPMENT.md                # This file (technical documentation)
```

### File Purposes

| File | Purpose | Key Concepts |
|------|---------|--------------|
| `main.py` | Entry point | Logging setup, bot startup |
| `src/config.py` | Config management | Pydantic settings, env vars |
| `src/agent/graph.py` | Agent logic | StateGraph, nodes, edges |
| `src/bot/telegram_bot.py` | User interface | Message handlers, media processing |
| `src/mcp/client.py` | Tool integration | MCP protocol, multi-server |
| `src/models/qwen_omni.py` | Model interface | BaseChatModel, multimodal encoding |

---

## Configuration

### Environment Variables (.env)

```bash
# Required
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...           # From @BotFather

# Model Configuration
MODEL_NAME=Qwen/Qwen3-Omni                     # HuggingFace model ID
MODEL_API_BASE=http://localhost:8000/v1        # API endpoint (or GPU IP)
MODEL_API_KEY=optional_api_key                 # If API requires auth

# LLM Parameters
MAX_TOKENS=2048                                # Max generation length
TEMPERATURE=0.7                                # Sampling temperature
TOP_P=0.9                                      # Nucleus sampling

# Application
LOG_LEVEL=INFO                                 # DEBUG, INFO, WARNING, ERROR
MAX_CONVERSATION_HISTORY=10                    # Messages to keep in memory
ENABLE_MEMORY=true                             # Enable conversation memory

# GPU
CUDA_VISIBLE_DEVICES=0                         # GPU device ID
TORCH_DTYPE=float16                            # float16, bfloat16, float32
```

### MCP Servers (config/mcp_servers.json)

```json
{
  "servers": {
    "server_name": {
      "command": "executable",
      "args": ["arg1", "arg2"],
      "transport": "stdio|streamable_http|sse",
      "url": "http://...",              // For HTTP/SSE
      "headers": {"Key": "Value"},      // For HTTP/SSE with auth
      "description": "Human description"
    }
  }
}
```

**Examples**:
- **Filesystem**: `npx -y @modelcontextprotocol/server-filesystem /path`
- **HTTP API**: Custom server at `http://localhost:8001/mcp`
- **Database**: Custom MCP server for DB operations

---

## Design Patterns

### 1. Factory Pattern

**Location**: `src/agent/graph.py:create_agent()`

**Purpose**: Centralize agent creation with configurable options

```python
agent = await create_agent(
    custom_tools=[my_tool],
    use_mcp=True,
    system_message="Custom prompt"
)
```

### 2. Singleton Pattern

**Location**: `src/mcp/client.py:get_mcp_client()`

**Purpose**: Single MCP client instance across application

```python
client = await get_mcp_client()  # Always returns same instance
```

### 3. Async Context Manager

**Location**: `src/mcp/client.py:MCPClientManager`

**Purpose**: Automatic resource management

```python
async with MCPClientManager() as client:
    tools = await client.get_tools()
# Automatic cleanup on exit
```

### 4. Dependency Injection

**Location**: `src/agent/graph.py:MultimodalAgent.__init__()`

**Purpose**: Decouple model and tools from agent logic

```python
agent = MultimodalAgent(
    model=qwen_model,
    tools=mcp_tools,
    system_message=prompt
)
```

### 5. Strategy Pattern

**Location**: `src/models/qwen_omni.py:_generate_local()` vs `_generate_api()`

**Purpose**: Switch between local and API inference transparently

---

## Key Decisions

### 1. Why Custom Qwen3 Omni Wrapper?

**Problem**: Qwen3 Omni not natively supported in LangChain

**Solution**: Implement `BaseChatModel` wrapper

**Benefits**:
- Full LangChain compatibility
- Custom multimodal message handling
- Dual inference modes (local/API)
- GPU optimization control

### 2. Why Separate Agent and Bot Layers?

**Problem**: Coupling UI and business logic limits flexibility

**Solution**: Clean separation with defined interfaces

**Benefits**:
- Agent can be reused with different frontends (CLI, web, etc.)
- Bot can be tested independently
- Easier to swap model or tools
- Clear responsibility boundaries

### 3. Why LangGraph Over LangChain Agents?

**Problem**: Traditional agents lack fine-grained control

**Solution**: Use LangGraph StateGraph for explicit workflow

**Benefits**:
- Deterministic execution flow
- Easy debugging (visualize graph)
- State introspection
- Conditional logic control
- Better error handling

**Reference**: [LangGraph v1.x patterns](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)

### 4. Why MCP Over Direct Tool Definitions?

**Problem**: Each tool source needs custom integration

**Solution**: Standardize on MCP protocol

**Benefits**:
- Access to existing MCP ecosystem (250+ servers)
- Standardized JSON-RPC 2.0 interface
- Multi-server support out of the box
- Easy to add new tool sources
- Transport flexibility (stdio, HTTP, SSE)

**Reference**: [LangChain MCP Integration](https://github.com/langchain-ai/langchain-mcp-adapters)

### 5. Why Telegram Over Web Interface?

**Problem**: Web interfaces require additional development

**Solution**: Leverage Telegram's existing infrastructure

**Benefits**:
- Native multimodal support
- No need to build file upload UI
- Mobile-friendly by default
- Push notifications built-in
- User authentication handled by Telegram
- Familiar UX for users

### 6. Base64 Encoding for Media

**Decision**: Encode images/audio/video as base64 in messages

**Rationale**:
- Self-contained messages (no external file references)
- Compatible with API-based inference
- Simplifies state management
- Works with LangChain message format

**Trade-off**: Larger message sizes (acceptable for GPU memory)

### 7. Conversation History Limiting

**Decision**: Keep only N recent messages (default: 10)

**Rationale**:
- Prevent context overflow
- Control GPU memory usage
- Maintain relevance (older context less useful)
- Configurable via `MAX_CONVERSATION_HISTORY`

**Implementation**: `src/agent/graph.py:_agent_node()` line 85

### 8. Temporary File Management

**Decision**: Store temp files per user, cleanup on /clear and shutdown

**Rationale**:
- Audio/video can't be efficiently base64 encoded for local inference
- User-scoped prevents cross-contamination
- Manual /clear gives user control
- Automatic shutdown cleanup prevents disk bloat

**Implementation**: `src/bot/telegram_bot.py:temp_files` dict

---

## Integration Points

### 1. LangGraph ↔ LangChain

**Interface**: `langchain_core.tools.BaseTool`

**Flow**:
1. MCP tools converted to `BaseTool` instances
2. Tools bound to model: `model.bind_tools(tools)`
3. LangGraph `ToolNode` executes tools
4. Results passed back to agent node

**Key Code**: `src/agent/graph.py:model_with_tools`

### 2. Telegram ↔ Agent

**Interface**: `langchain_core.messages.BaseMessage`

**Flow**:
1. Telegram update → Media download
2. Create `HumanMessage` with multimodal content
3. Agent processes and returns `AIMessage`
4. Extract text from response
5. Send to Telegram user

**Key Code**: `src/bot/telegram_bot.py:handle_message()`

### 3. MCP Client ↔ LangGraph

**Interface**: `langchain_mcp_adapters.tools.load_mcp_tools()`

**Flow**:
1. MCPClientManager connects to servers
2. `MultiServerMCPClient.get_tools()` fetches all tools
3. Tools passed to `create_agent()`
4. Agent binds tools to model
5. LangGraph routes tool calls to ToolNode

**Key Code**: `src/agent/graph.py:create_agent()`

### 4. Model ↔ Multimodal Inputs

**Interface**: Custom content format (list of dicts)

**Flow**:
1. `create_multimodal_message()` encodes media
2. Message with `content: List[Dict]` created
3. Model wrapper processes content list
4. Processor handles text + media
5. Model generates response

**Key Code**: `src/models/qwen_omni.py:_format_message_content()`

---

## Deployment Strategy

### Local Development

**Target**: Developer workstation with NVIDIA GPU

**Steps**:
1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env && nano .env`
4. `python main.py`

**Use Case**: Development, debugging, testing

### Docker Deployment

**Target**: Any system with Docker + NVIDIA Container Toolkit

**Steps**:
1. `cp .env.example .env && nano .env`
2. `docker-compose up -d --build`
3. `docker-compose logs -f`

**Benefits**:
- Consistent environment
- Easy updates (`docker-compose pull && docker-compose up -d`)
- Resource isolation
- GPU access via NVIDIA runtime

**Configuration**: `docker-compose.yml`

### DigitalOcean GPU Droplet

**Target**: Production deployment on cloud GPU

**Steps**:
1. Create GPU Droplet (NVIDIA Tesla, A100, etc.)
2. SSH into droplet
3. `git clone <repo> && cd multimodal`
4. `chmod +x deploy/digitalocean_setup.sh`
5. `./deploy/digitalocean_setup.sh`
6. Edit `.env` with bot token
7. `docker-compose up -d --build`

**Script**: `deploy/digitalocean_setup.sh`
- Installs NVIDIA drivers
- Installs Docker + NVIDIA Container Toolkit
- Sets up environment
- Creates directory structure

### Systemd Service

**Target**: Auto-start on server boot

**Steps**:
1. `chmod +x deploy/systemd_service.sh`
2. `./deploy/systemd_service.sh`
3. `sudo systemctl enable multimodal-agent`
4. `sudo systemctl start multimodal-agent`

**Benefits**:
- Automatic restart on failure
- Starts on system boot
- Centralized logging via journalctl

**Script**: `deploy/systemd_service.sh`

---

## Testing Strategy

### Unit Tests

**Location**: `tests/`

**Coverage**:
- `test_agent.py`: Agent creation, message processing
- `test_mcp.py`: MCP client initialization, tool loading

**Framework**: pytest with pytest-asyncio

**Run**: `pytest tests/`

### Integration Testing

**Manual Testing Checklist**:
- [ ] Text message → response
- [ ] Image + caption → analysis
- [ ] Audio file → transcription
- [ ] Video file → summary
- [ ] /start command → welcome message
- [ ] /help command → help text
- [ ] /clear command → history cleared
- [ ] Tool call → correct execution
- [ ] Error handling → user-friendly message

### Performance Testing

**Metrics to Monitor**:
- Response latency (time to first token)
- GPU memory usage
- Conversation context size
- Temp file cleanup

**Tools**:
- `nvidia-smi` for GPU monitoring
- `docker stats` for container resources
- Application logs for timing

---

## Future Considerations

### 1. Scalability

**Current**: Single-instance bot

**Future Options**:
- Multiple Telegram bot instances (load balancing)
- Queue-based processing (Celery, Redis)
- Separate model server (vLLM, TGI)
- Distributed MCP servers

### 2. Memory/Personalization

**Current**: In-memory conversation history (limited)

**Future Options**:
- Persistent storage (PostgreSQL, Redis)
- User profiles and preferences
- Long-term memory (vector DB)
- LangGraph checkpointing for state persistence

**Hint**: LangGraph supports memory via `MemorySaver` - can be added to `graph.compile(checkpointer=memory)`

### 3. Advanced Features

**Potential Additions**:
- Voice message → voice response (TTS)
- Image generation tools
- Multi-turn tool workflows
- Human-in-the-loop approval for sensitive tools
- Usage analytics and monitoring
- Rate limiting per user

### 4. Monitoring & Observability

**Current**: Loguru file logging

**Future Options**:
- LangSmith integration (LangChain tracing)
- Prometheus metrics export
- Grafana dashboards
- Error tracking (Sentry)
- Performance profiling

### 5. Security Enhancements

**Considerations**:
- User authentication/authorization
- Rate limiting (prevent abuse)
- Input sanitization (prevent injection)
- Secure MCP server connections (TLS)
- Secret management (Vault, AWS Secrets Manager)
- Audit logging

### 6. Model Improvements

**Options**:
- Model quantization (GPTQ, AWQ) for efficiency
- LoRA adapters for domain specialization
- Ensemble models for better quality
- Fallback models for reliability

### 7. Additional Frontends

**Beyond Telegram**:
- Web interface (FastAPI + WebSocket)
- CLI interface (for developers)
- Slack bot
- Discord bot
- REST API for programmatic access

---

## Troubleshooting Guide

### Common Issues

#### 1. "ImportError: No module named 'langchain'"

**Cause**: Dependencies not installed

**Solution**: `pip install -r requirements.txt`

#### 2. "CUDA out of memory"

**Cause**: GPU memory exhausted

**Solutions**:
- Reduce `MAX_TOKENS`
- Use `TORCH_DTYPE=float16` instead of float32
- Decrease `MAX_CONVERSATION_HISTORY`
- Use API-based inference instead of local

#### 3. "Telegram bot not responding"

**Cause**: Invalid bot token or network issues

**Solutions**:
- Verify `TELEGRAM_BOT_TOKEN` in `.env`
- Check bot is running: `docker-compose ps`
- Review logs: `docker-compose logs -f`
- Test bot token: `curl https://api.telegram.org/bot<TOKEN>/getMe`

#### 4. "MCP tools not loading"

**Cause**: MCP server configuration issues

**Solutions**:
- Validate JSON syntax in `config/mcp_servers.json`
- Check server executables are installed (e.g., `npx`)
- Verify HTTP endpoints are accessible
- Review logs for connection errors

#### 5. "ModuleNotFoundError: No module named 'src'"

**Cause**: Python path issues

**Solution**: Run from project root: `cd /path/to/multimodal && python main.py`

---

## Development Workflow

### Adding a New Custom Tool

1. **Create Tool File**: `src/tools/my_tool.py`

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Description of what this tool does."""
    # Implementation
    return result
```

2. **Import in Agent Creation**:

```python
from src.tools.my_tool import my_custom_tool

agent = await create_agent(
    custom_tools=[my_custom_tool],
    use_mcp=True
)
```

3. **Test**: Send message to bot that would trigger tool usage

### Adding a New MCP Server

1. **Edit** `config/mcp_servers.json`:

```json
{
  "servers": {
    "my_server": {
      "transport": "streamable_http",
      "url": "http://my-server:8000/mcp",
      "headers": {"Authorization": "Bearer TOKEN"},
      "description": "My custom MCP server"
    }
  }
}
```

2. **Restart**: `docker-compose restart` or re-run `python main.py`

3. **Verify**: Check logs for "Loaded X MCP tools"

### Modifying the Agent Workflow

**Example**: Add a preprocessing node

1. **Edit** `src/agent/graph.py`:

```python
def _preprocess_node(state: AgentState) -> dict:
    """Preprocess user input."""
    messages = state["messages"]
    # Custom preprocessing logic
    return {"messages": messages}

# In _build_graph():
workflow.add_node("preprocess", self._preprocess_node)
workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "agent")
```

2. **Test**: Verify new node in execution flow

---

## Performance Benchmarks

### Expected Latency (RTX 4090, float16)

| Task | Latency | Notes |
|------|---------|-------|
| Text (100 tokens) | ~1-2s | Depends on prompt length |
| Image analysis | ~2-3s | Single image |
| Audio transcription | ~3-5s | 10s audio file |
| Video summary | ~10-15s | 30s video clip |

### Memory Usage

| Configuration | VRAM | RAM |
|---------------|------|-----|
| float16 | ~12-16 GB | ~8 GB |
| float32 | ~24-32 GB | ~12 GB |
| 4-bit quantized | ~6-8 GB | ~6 GB |

### Scaling

| Metric | Single Instance | Notes |
|--------|----------------|-------|
| Concurrent users | 5-10 | Depends on GPU |
| Requests/min | 20-30 | With queueing |
| Max context | 8K tokens | Model dependent |

---

## Code Quality Standards

### Python Style

- **Formatter**: Black (line length: 88)
- **Linter**: Ruff
- **Type Hints**: Encouraged, not enforced
- **Docstrings**: Google style

### Commit Conventions

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### Code Review Checklist

- [ ] Code follows Black formatting
- [ ] No linter warnings (Ruff)
- [ ] Docstrings for public functions
- [ ] Error handling present
- [ ] Logs added for debugging
- [ ] Tests added/updated
- [ ] README/DEVELOPMENT.md updated if needed

---

## Dependencies Rationale

### Why These Specific Versions?

| Package | Version | Reason |
|---------|---------|--------|
| langchain | >=0.3.0 | Latest features, MCP support |
| langgraph | >=0.2.0 | StateGraph improvements |
| langchain-mcp-adapters | >=0.1.0 | Official MCP integration |
| python-telegram-bot | >=21.0 | Async support, latest API |
| transformers | >=4.45.0 | Qwen3 model support |
| torch | >=2.0.0 | CUDA 12+ support |
| vllm | >=0.6.0 | Efficient inference |

### Optional Dependencies

**Not Included** (add if needed):
- `langsmith`: LangChain tracing and debugging
- `ray`: Distributed computing
- `redis`: Caching and queuing
- `sqlalchemy`: Database ORM
- `chromadb`: Vector database for memory

---

## References & Resources

### Official Documentation

1. **LangGraph Multi-Agent Tutorial**
   https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
   - StateGraph patterns
   - Node and edge definitions
   - Multi-agent orchestration

2. **LangChain MCP Adapters**
   https://github.com/langchain-ai/langchain-mcp-adapters
   - Installation and setup
   - MultiServerMCPClient usage
   - Transport configuration

3. **Build Multimodal Agents (Google Cloud)**
   https://cloud.google.com/blog/products/ai-machine-learning/build-multimodal-agents-using-gemini-langchain-and-langgraph
   - Multimodal agent architecture
   - Image/audio/video processing
   - Agent specialization patterns

4. **LangGraph MCP Integration Guide**
   https://latenode.com/blog/langgraph-mcp-integration-complete-model-context-protocol-setup-guide-working-examples-2025
   - Complete setup guide
   - Working examples
   - Best practices

5. **Model Context Protocol**
   https://modelcontextprotocol.io/
   - Protocol specification
   - Available MCP servers
   - Building custom servers

### Community Resources

- LangChain Discord: https://discord.gg/langchain
- LangGraph GitHub: https://github.com/langchain-ai/langgraph
- Telegram Bot API: https://core.telegram.org/bots/api
- Qwen Models: https://github.com/QwenLM/Qwen

---

## Changelog

### v0.1.0 (2025-11-22)

**Initial Release**

- ✅ LangGraph agent with StateGraph workflow
- ✅ Qwen3 Omni multimodal model integration
- ✅ MCP server tool integration
- ✅ Telegram bot frontend
- ✅ Docker deployment with GPU support
- ✅ DigitalOcean setup scripts
- ✅ Systemd service configuration
- ✅ Comprehensive documentation
- ✅ Unit tests for core components

---

## Contact & Support

**For Questions**:
- Check README.md for usage documentation
- Review this file for technical details
- Check GitHub issues (if public repo)

**For Contributions**:
- Fork repository
- Create feature branch
- Follow code quality standards
- Submit pull request with clear description

---

**Last Updated**: 2025-11-22
**Maintainer**: [Your Name/Team]
**License**: [Your License]
