import operator
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Loggings
import logging
import sys

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

SYSTEM_PROMPT = """
You are a concise weather assistant for locations in the United States.

Tool usage:
- If the user gives a city or place name, call get_location first.
- Use the returned latitude and longitude with get_forecast.
- Use the returned state_code with get_alerts when weather alerts are relevant.
- For a general weather request such as "What's the weather in San Francisco?",
  retrieve both the forecast and active alerts.
- If the user already provides latitude and longitude, you can call
  get_forecast directly.
- If the user explicitly asks for alerts, call get_alerts directly.
- Never invent coordinates or state codes.

When answering:
- Start directly with useful information.
- Use clean Markdown.
- Summarize forecast and important alerts clearly.
- End with a short takeaway when appropriate.
- Do not mention MCP internals.
"""

# LLM configuration
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    # max_completion_tokens=256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MCP_SERVER_PATH = os.getenv(
    "MCP_SERVER_PATH",
    "server/weather.py",
)

server_params = StdioServerParameters(
    command=sys.executable,
    args=[
        str(PROJECT_ROOT / MCP_SERVER_PATH),
    ],
)
# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # Keeps debugging information about MCP calls
    tool_trace: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------


async def build_graph(session: ClientSession):

    # Discover available MCP tools
    tools_response = await session.list_tools()

    mcp_tools = tools_response.tools

    logger.info(
        "Available MCP tools: %s",
        [tool.name for tool in mcp_tools],
    )

    # Convert MCP tool schema -> LLM tool schema
    llm_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in mcp_tools
    ]

    llm_with_tools = llm.bind_tools(llm_tools)

    # -----------------------------------------------------
    # LLM node
    # -----------------------------------------------------

    async def chatbot(state: AgentState):

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]

        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response]}

    # -----------------------------------------------------
    # MCP tools node
    # -----------------------------------------------------

    async def call_mcp_tools(state: AgentState):

        last_message = state["messages"][-1]

        tool_messages = []
        tool_trace = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # Keep terminal debugging too
            logger.info(
                "Calling MCP tool=%s arguments=%s",
                tool_name,
                tool_args,
            )

            result = await session.call_tool(
                tool_name,
                arguments=tool_args,
            )

            text_blocks = [block.text for block in result.content if isinstance(block, TextContent)]

            tool_output = "\n".join(text_blocks)

            if not tool_output:
                tool_output = "Tool returned no text content."

            tool_messages.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call_id,
                )
            )

            # Information that Streamlit can display
            tool_trace.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                    "is_error": bool(getattr(result, "isError", False)),
                    "output": tool_output,
                }
            )

        return {
            "messages": tool_messages,
            "tool_trace": tool_trace,
        }

    # -----------------------------------------------------
    # Graph definition
    # -----------------------------------------------------

    builder = StateGraph(AgentState)

    builder.add_node(
        "chatbot",
        chatbot,
    )

    builder.add_node(
        "tools",
        call_mcp_tools,
    )

    builder.add_edge(
        START,
        "chatbot",
    )

    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    builder.add_edge(
        "tools",
        "chatbot",
    )

    return builder.compile()


# ---------------------------------------------------------
# Public function used by Streamlit / CLI
# ---------------------------------------------------------

# ReAct loop: chatbot → tools_condition → MCP tool → chatbot
import time


async def run_agent(
    history: list[dict[str, str]],
    query: str,
) -> dict[str, Any]:

    messages: list[BaseMessage] = []

    # Convert UI history -> LangChain messages
    for message in history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))

        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))

    # Current request
    messages.append(HumanMessage(content=query))

    # ΤΙΜΕ
    total_start = time.perf_counter()
    mcp_start = time.perf_counter()

    # Start/connect to MCP server
    async with stdio_client(server_params) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()
            # ΤΙΜΕ
            logger.info(
                "MCP startup completed in %.2fs",
                time.perf_counter() - mcp_start,
            )

            graph_start = time.perf_counter()
            graph = await build_graph(session)

            logger.info(
                "Graph build completed in %.2fs",
                time.perf_counter() - graph_start,
            )

            agent_start = time.perf_counter()

            result = await graph.ainvoke(
                {
                    "messages": messages,
                    "tool_trace": [],
                },
                config={
                    # Protection against infinite agent loops
                    "recursion_limit": 10,
                },
            )

            logger.info(
                "Agent execution completed in %.2fs",
                time.perf_counter() - agent_start,
            )

            logger.info(
                "Request completed in %.2fs",
                time.perf_counter() - total_start,
            )

            return {
                "answer": result["messages"][-1].content,
                "tool_trace": result.get(
                    "tool_trace",
                    [],
                ),
            }
