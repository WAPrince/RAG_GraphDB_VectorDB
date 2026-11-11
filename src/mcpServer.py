import asyncio
import uvicorn
import inspect
from fastmcp import FastMCP
from retrieval.vectorDbRetrieval import  search_documents_in_vectordb
from retrieval.graphDbRetrieval import search_documents_in_graphdb

mcp = FastMCP()
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))

@mcp.tool()
def search_in_vectordb(query: str, limit: int = 3) -> str:
    """Search the qdrant vectordb ( knowledge base )  for information relevant to the query."""

    try:
        documents = search_documents_in_vectordb(query, limit=limit)
    except Exception as e:
        print(f"Exception {e}")
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

    # print(f"search_in_graphdb LLM delivered search string: {query}") # "AMD"

    # Text-to-Cypher Ansätze:
    #
    # 1. Der GraphRAG Ansatz (Fortgeschritten & Sicher)
    # •  Schema auslesen: Das Framework zieht dynamisch die Struktur aus Neo4j (apoc.meta.schema()).
    # •  Cypher-Generierung: Das LLM erzeugt die Query basierend auf dem dynamischen Schema.
    # •  Validierung & Ausführung: Ein Abfang-Skript prüft die Query auf Syntaxfehler, bevor sie an Neo4j geschickt wird.
    # •  Antwort-Synthese: Das LLM erhält die Rohdaten aus Neo4j und formuliert einen Satz für den Nutzer.
    # ...

    # 2. Der semantische Layer / Werkzeug-Ansatz (Robust für Produktion)
    # In produktiven Systemen lässt man das LLM oft kein freies Cypher schreiben, da das Risiko von Fehlern oder unerwünschten Datenbank-Injektionen hoch ist. Stattdessen nutzt man einen semantischen Layer via Function Calling:
    # • Vorgefertigte Funktionen: Entwickler schreiben sichere, parametrisierte Cypher-Abfragen (z. B. get_movies_by_actor(actor_name)).
    # • LLM als Router: Das LLM entscheidet anhand der Nutzerfrage nur, welche Funktion aufgerufen werden muss und extrahiert die Argumente (z. B. actor_name = "Keanu Reeves").
    # • Ausführung: Die Anwendung führt die sichere Funktion in Neo4j aus.
    generated_cypher_test = """    
            MATCH (amd {id: 'AMD_v2'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v3'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v4'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v5'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v6'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v7'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v8'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v9'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten
          """.strip()
    parameters = {}

    return search_documents_in_graphdb(generated_cypher=generated_cypher_test, parameters=parameters)


async def main():
    print("Starte MCP-Server über HTTP...")

    # run_http_async startet einen uvicorn-Server im Hintergrund.
    # Standardmäßig wird der moderne "streamable-http"-Transport verwendet.
    await mcp.run_http_async(
        host="127.0.0.1",
        port=3005,
        transport="streamable-http"  # Optionen: "streamable-http", "sse" oder "http"
    )


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["mcp.server.streamable_http"] = {"level": "CRITICAL"}
    asyncio.run(main())