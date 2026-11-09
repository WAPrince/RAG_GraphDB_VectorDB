import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agent.bedrock import create_bedrock_llm
from graph.state import AgentState
from mcpapi.mcpClient import create_mcp_client


async def run_agent(prompt: str) -> None:

    load_dotenv()
    print(f" AWS_REGION: {os.getenv("AWS_REGION")}")
    print(f" AWS_PROFILE: {os.getenv("AWS_PROFILE")}")

    MAX_TOOL_CALLS = 3
    mcp_url = "http://localhost:3005/mcp"

    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            llm = create_bedrock_llm()
            mcp_client = create_mcp_client()
            mcp_tools = await mcp_client.get_tools()
            tools = [
                *mcp_tools,
            ]
            llm_with_tools = llm.bind_tools(tools)

            def call_model(state: AgentState) -> dict:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            async def tool_middleware_node(state: AgentState):
                current_count = state.get("tool_call_count", 0)
                new_count = current_count + 1
                print(f"\n[Middleware] Tool-Aufruf {new_count} von maximal {MAX_TOOL_CALLS} wird ausgeführt...")
                tool_node_instance = ToolNode(mcp_tools)
                result = await tool_node_instance.ainvoke(state)
                return {**result, "tool_call_count": new_count}

            def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
                last_message = state["messages"][-1]
                current_count = state.get("tool_call_count", 0)
                if last_message.tool_calls:
                    if current_count >= MAX_TOOL_CALLS:
                        print(
                            f"\n[Middleware] HINWEIS: Limit von {MAX_TOOL_CALLS} Tool-Aufrufen erreicht! Breche ab.")
                        return "__end__"
                    return "tools"
                return "__end__"

            builder = StateGraph(AgentState)
            builder.add_node("call_model", call_model)
            builder.add_node("tools", tool_middleware_node)

            builder.add_edge(START, "call_model")
            builder.add_conditional_edges("call_model", should_continue)
            builder.add_edge("tools", "call_model")

            checkpointer = InMemorySaver()
            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    "thread_id": "console-session-1"
                }
            }
            result = await graph.ainvoke(
                {
                    "messages": [
                        ("user", prompt)
                    ]
                },
                config=config,
            )
            messages = result.get("messages", [])
            if messages:
                print(messages[-1].content)


async def main() -> None:
    await run_agent(
        "Was ist den Piloten passiert? Gib deine Quellen an!")


if __name__ == "__main__":
    asyncio.run(main())
