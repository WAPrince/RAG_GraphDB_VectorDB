from mcp.server.fastmcp import FastMCP
import inspect
from retrieval.vectorDbRetrieval import  search_documents_in_vectordb
from retrieval.graphDbRetrieval import search_documents_in_graphdb

mcp = FastMCP("demo-server",port=3005)
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))

@mcp.tool()
def search_in_vectordb(query: str, limit: int = 3) -> str:
    """Search the qdrant vectordb ( knowledge base )  for information relevant to the query."""

    print("vor")
    try:
        documents = search_documents_in_vectordb(query, limit=limit)
    except Exception as e:
        print(f"Exception {e}")
    print("nach")
    if not documents:
        return "No relevant documents found."##
    result = []
    for document in documents:
        result.append(
            f"""
    Source: {document["source"]}
    Similarity: {document["score"]:.4f}
    {document["text"]}
    #""".strip())
    return "\n\n---\n\n".join(result)

@mcp.tool()
def search_in_graphdb(query: str) -> str:
    """Search the neo graph db for information relevant to the query."""

    print(f"search_in_graphdb LLM delivered search string: {query}") # "AMD"
    # TODO transform LLM query to NEO Graph query
    generated_cypher_test = """    
            MATCH (amd {id: 'AMD_v2'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v3'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v4'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v5'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v6'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v7'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v8'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v9'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten
          """.strip()
    parameters = {}

    return search_documents_in_graphdb(generated_cypher=generated_cypher_test, parameters=parameters)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
