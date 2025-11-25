"""LangGraph agent with multimodal support and tool calling."""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger

from ..config import settings
from ..mcp import get_all_mcp_tools
from ..models import Qwen3OmniModel


class AgentState(MessagesState):
    """
    State for the multimodal agent graph.

    Extends MessagesState with additional user tracking.

    Attributes:
        messages: Conversation message history (inherited from MessagesState)
        user_id: User identifier (for memory/personalization)
    """

    user_id: str


class MultimodalAgent:
    """
    Bi-directional multimodal AI agent with tool calling and MCP support.

    Features:
    - Multimodal input/output (text, images, audio, video)
    - Tool calling via MCP servers
    - Conversation memory
    - State management with LangGraph
    """

    def __init__(
        self,
        model: Qwen3OmniModel,
        tools: List[BaseTool],
        system_message: str = None,
    ):
        """
        Initialize the multimodal agent.

        Args:
            model: Qwen3 Omni model instance
            tools: List of tools (from MCP or custom)
            system_message: System prompt for the agent
        """
        self.model = model
        self.tools = tools
        self.system_message = system_message or self._default_system_message()

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools) if self.tools else self.model

        # Build the graph
        self.graph = self._build_graph()

    def _default_system_message(self) -> str:
        """Default system message for the agent."""
        return """You are a helpful multimodal AI assistant powered by Qwen3 Omni.

You can:
- Process and understand text, images, audio, and video
- Use various tools to help users accomplish tasks
- Access external systems via MCP servers
- Maintain context across conversations

When responding:
- Be helpful, accurate, and concise
- Ask clarifying questions when needed
- Use tools when appropriate to provide better assistance
- Explain your reasoning when using tools

Always prioritize user safety and provide accurate information."""

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Graph structure:
        1. agent: Main reasoning node (calls model)
        2. tools: Tool execution node (if tools are invoked)
        3. Conditional edges based on whether tools are called
        """
        # Initialize graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)

        if self.tools:
            workflow.add_node("tools", ToolNode(self.tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        if self.tools:
            # If agent calls tools, go to tools node; otherwise end
            workflow.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "tools",
                    END: END,
                },
            )
            # After tools, go back to agent
            workflow.add_edge("tools", "agent")
        else:
            # No tools, just end after agent
            workflow.add_edge("agent", END)

        # Compile graph
        return workflow.compile()

    def _agent_node(self, state: AgentState) -> dict:
        """
        Agent reasoning node.

        Args:
            state: Current agent state

        Returns:
            Updated state with new messages
        """
        messages = state["messages"]

        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_message)] + list(messages)

        # Limit conversation history
        if len(messages) > settings.max_conversation_history + 1:  # +1 for system message
            messages = [messages[0]] + list(messages[-(settings.max_conversation_history):])

        # Invoke model
        logger.info(f"Agent processing {len(messages)} messages")
        response = self.model_with_tools.invoke(messages)

        return {"messages": [response]}

    async def ainvoke(self, user_message: BaseMessage, user_id: str = "default") -> BaseMessage:
        """
        Process a user message asynchronously.

        Args:
            user_message: User's message (can be multimodal)
            user_id: User identifier for state management

        Returns:
            Agent's response message
        """
        # Create initial state
        initial_state = {
            "messages": [user_message],
            "user_id": user_id,
        }

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        # Get the last message (agent's response)
        response = result["messages"][-1]

        return response

    def invoke(self, user_message: BaseMessage, user_id: str = "default") -> BaseMessage:
        """
        Process a user message synchronously.

        Args:
            user_message: User's message (can be multimodal)
            user_id: User identifier for state management

        Returns:
            Agent's response message
        """
        # Create initial state
        initial_state = {
            "messages": [user_message],
            "user_id": user_id,
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Get the last message (agent's response)
        response = result["messages"][-1]

        return response

    async def astream(self, user_message: BaseMessage, user_id: str = "default"):
        """
        Stream agent responses.

        Args:
            user_message: User's message (can be multimodal)
            user_id: User identifier for state management

        Yields:
            Incremental updates from the agent
        """
        initial_state = {
            "messages": [user_message],
            "user_id": user_id,
        }

        async for chunk in self.graph.astream(initial_state):
            yield chunk


async def create_agent(
    custom_tools: List[BaseTool] = None,
    use_mcp: bool = True,
    system_message: str = None,
) -> MultimodalAgent:
    """
    Factory function to create a configured multimodal agent.

    Args:
        custom_tools: Additional custom tools to include
        use_mcp: Whether to load MCP tools
        system_message: Custom system message

    Returns:
        Configured MultimodalAgent instance
    """
    # Initialize model
    model = Qwen3OmniModel()

    # Gather tools
    tools = custom_tools or []

    if use_mcp:
        logger.info("Loading MCP tools...")
        mcp_tools = await get_all_mcp_tools()
        tools.extend(mcp_tools)
        logger.info(f"Loaded {len(mcp_tools)} MCP tools")

    logger.info(f"Creating agent with {len(tools)} total tools")

    # Create agent
    agent = MultimodalAgent(
        model=model,
        tools=tools,
        system_message=system_message,
    )

    return agent
