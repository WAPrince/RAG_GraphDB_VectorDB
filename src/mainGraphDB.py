import asyncio

from langgraphConfig.graph import create_graph

async def run_agent(prompt: str) -> None:
    graph = await create_graph()
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
        # GraphDB
        "Was kannst du zu AMD sagen? Verwende Daten aus Vektor- und Graphen-Datenbank!")
        #"Was kannst du zu AMD sagen?")

if __name__ == "__main__":
    asyncio.run(main())