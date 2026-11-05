from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.bedrock import create_bedrock_llm
from langgraphConfig.state import AgentState
from mcpapi.mcpClient import create_mcp_client


async def create_graph():
    llm = create_bedrock_llm()

    mcp_client = create_mcp_client()
    mcp_tools = await mcp_client.get_tools()
    tools = [
        *mcp_tools,
    ]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,
    )

    def call_model(state: AgentState) -> dict:
        #print(f"Call_model DAVOR")
        response = llm_with_tools.invoke(state["messages"])
        #print(f"Call_model DANACH")
        return {"messages": [response]}

    def route_after_model(state: AgentState) -> str:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return "end"
        return "tools"

    ###################################################

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "end": END,
            "tools": "tools",
       },
    )
    builder.add_edge("tools", "call_model")
    checkpointer = InMemorySaver()
    return builder.compile(
        checkpointer=checkpointer
    )