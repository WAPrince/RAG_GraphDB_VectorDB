import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from graph.state import AgentState
from mcpapi.mcpClient import MCP_SERVERS


async def run_agent(prompt: str) -> None:

    mcp_url = "http://localhost:3005/mcp"
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            model = init_chat_model(
                model="claude-sonnet-4-6",
                model_provider="anthropic"
            )
            MAX_TOOL_CALLS = 3

            client = MultiServerMCPClient(MCP_SERVERS)
            async with client.session("knowledge") as session:
                mcp_tools = await load_mcp_tools(session)
                model_with_tools = model.bind_tools(mcp_tools)

                # 2. Nodes definieren
                async def call_model(state: AgentState):
                    response = await model_with_tools.ainvoke(state["messages"])
                    # response = await model.ainvoke(state["messages"])
                    return {"messages": [response]}

                # Middleware-Funktion vor der Tool-Ausführung: Erhöht den Zähler
                async def tool_middleware_node(state: AgentState):
                    # Wir holen den aktuellen Zählerstand (Default 0, falls noch nicht gesetzt)
                    current_count = state.get("tool_call_count", 0)
                    new_count = current_count + 1
                    print(f"\n[Middleware] Tool-Aufruf {new_count} von maximal {MAX_TOOL_CALLS} wird ausgeführt...")

                    # Leitet den Aufruf an den echten ToolNode weiter
                    tool_node_instance = ToolNode(mcp_tools)
                    result = await tool_node_instance.ainvoke(state)

                    # Wir geben das Ergebnis des Tools UND den aktualisierten Zähler zurück
                    return {**result, "tool_call_count": new_count}

                # Bedingte Logik mit Limit-Prüfung
                def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
                    last_message = state["messages"][-1]
                    current_count = state.get("tool_call_count", 0)

                    # Falls das Modell Tools aufrufen will
                    if last_message.tool_calls:
                        # HIER greift die Begrenzung: Wenn das Limit erreicht ist, brechen wir ab
                        if current_count >= MAX_TOOL_CALLS:
                            print(f"\n[Middleware] HINWEIS: Limit von {MAX_TOOL_CALLS} Tool-Aufrufen erreicht! Breche ab.")
                            return "__end__"
                        return "tools"
                    return "__end__"

                # Builder
                builder = StateGraph(AgentState)
                builder.add_node("call_model", call_model)
                builder.add_node("tools", tool_middleware_node)

                builder.add_edge(START, "call_model")
                builder.add_conditional_edges("call_model", should_continue)
                builder.add_edge("tools", "call_model")

                checkpointer = InMemorySaver()
                graph=builder.compile( checkpointer=checkpointer)

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
        "Was kannst du zu AMD sagen? Gib deine Quellen an!")

if __name__ == "__main__":
    asyncio.run(main())