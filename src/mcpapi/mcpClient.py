from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_SERVERS = {
    "knowledge": {
        "transport": "streamable-http",
        "url": "http://localhost:3005/mcp",
    }
}

def create_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(MCP_SERVERS)