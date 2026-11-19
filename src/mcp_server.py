import asyncio
from qdrant_client.models import ScoredPoint
import uvicorn
import inspect
import json
from fastmcp import FastMCP
from mds.rag.vectordb_access import qdrant_vector_search
from mds.rag.graphdb_access import neo_graph_search
from mds.rag.embeddings import create_embedding_model, BM25_MODEL_NAME,RERANKER_MODEL_NAME,SEMANTIC_SEARCH_EMBEDDING_MODEL
import pandas as pd
from sentence_transformers import CrossEncoder
import traceback


reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

mcp = FastMCP()
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))

def _merge_and_rerank(query: str, list_a: list[dict], list_b: list[dict], top_k: int = 3) -> list[dict]:

    # 1. Qdrant-Ergebnisse sicher in Dictionaries umwandeln
    vector_docs = []
    for hit in list_a:
        if isinstance(hit, dict):
            vector_docs.append(hit)
        else:  # Ist ein ScoredPoint-Objekt
            vector_docs.append({
                "id": str(hit.id),
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "source": "vector",
            })
    # 2. Neo4j-Ergebnisse sicher in Dictionaries umwandeln
    graph_docs = []
    for idx, row in enumerate(list_b):
        if isinstance(row, dict):
            graph_docs.append({
                "id": row.get("id", f"graph_{idx}"),
                "text": row.get(
                    "text",
                    f"Beziehung: {row.get('Beziehung', '')}, Zielknoten:"
                    f" {row.get('Zielknoten', '')}",
                ),
                "score": row.get("score", 1.0),
                "source": "graph",
            })
        else:  # Falls list_b doch noch ScoredPoint-Objekte enthält
            graph_docs.append({
                "id": str(row.id),
                "text": row.payload.get("text", ""),
                "score": row.score,
                "source": "graph",
            })

    reranked =[]
    try:
        # Schritt A: Ergebnisse zusammenführen & Duplikate anhand der ID entfernen
        unique_docs = {}
        for doc in vector_docs + graph_docs:
            doc_id = doc["id"]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
            else:
                # Optional: Vermerken, dass das Dokument in beiden Datenbanken gefunden wurde
                unique_docs[doc_id]["source"] = "both"

        candidates = list(unique_docs.values())
        if not candidates:
            return []
        # Schritt B: Paare aus (Suchanfrage, Dokumenttext) für den Cross-Encoder bauen
        pairs = [[query, doc["text"]] for doc in candidates]
        # Schritt C: Neue einheitliche Scores berechnen
        scores = reranker.predict(pairs)
        # Schritt D: Scores zuweisen und absteigend sortieren
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        traceback.print_exc()
    return json.dumps(reranked[:top_k], ensure_ascii=False, indent=2, default=float)

#@mcp.tool()
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

#@mcp.tool()
def search_in_graphdb(query: str) -> str:
    """Search the neo graph db for information relevant to the query."""

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


@mcp.tool()
def search_in_graphdb_and_vectordb(query:str, limit: int,top_k: int) -> str :

    hybrid=True
    result=None
    try:
        """Search the qdrant db for information relevant to the query."""
        vectordb_result:dict[Any, Any] =  qdrant_vector_search(query, limit, "semiconductor__dense_sparse",   SEMANTIC_SEARCH_EMBEDDING_MODEL,   hybrid )

        """Search the neo graph db for information relevant to the query."""
        generated_cypher_test = """
                 MATCH (amd {id: 'AMD_v2'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v3'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v4'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v5'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v6'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v7'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v8'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten UNION MATCH (amd {id: 'AMD_v9'})-[r]->(target) RETURN type(r) AS Beziehung, target.id AS Zielknoten
                  """.strip()
        parameters = {}
        graphdb_result_json:dict[Any, Any]=neo_graph_search(generated_cypher_test, parameters )
        graphdb_result_json_clean = graphdb_result_json.replace("\n", "")
        data = json.loads(graphdb_result_json_clean)
        graphdb_result_list = [dict(zip(rec["keys"], rec["_fields"])) for rec in data["records"]]
        graph_scored_points = [
            ScoredPoint(
                id=idx,  # Laufende Nummer als ID (z. B. 0, 1, ...)
                version=1,  # Standard-Version wie bei Qdrant
                score=1.0,  # Künstlicher Relevanz-Score (1.0 = exakter Treffer)
                payload={
                    # Hier packst du die Neo4j-Daten so rein, wie Qdrant sie erwartet
                    "text": f"Beziehung: {row['Beziehung']}, Zielknoten: {row['Zielknoten']}",
                    **row,  # Optional: Behält 'Beziehung' und 'Zielknoten' zusätzlich als eigene Keys im payload
                },
            )
            for idx, row in enumerate(graphdb_result_list)
        ]

        # merge nad rerank Qdrant and NEo results
        result:list[dict]=_merge_and_rerank(query=query, list_a=vectordb_result, list_b=graph_scored_points, top_k=3)

    except Exception as e:
        traceback.print_exc()
    return result

async def main():
    print("Starte MCP-Server über HTTP...")

    # run_http_async startet einen uvicorn-Server im Hintergrund.
    await mcp.run_http_async(
        host="127.0.0.1",
        port=3005,
        transport="streamable-http"
    )


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["mcp.server.streamable_http"] = {"level": "CRITICAL"}
    asyncio.run(main())