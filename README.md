
Features:

    1. Use a Langgraph graph

    2. Call either Anthropic directly or AWS Bedrock

    3. Call MCP tools at most 3 times 

**--------------------------------------------------------------------------------------------------------------------**

USE CASE 1 : via MCP query a VectorDB (Qdrant) and a GraphDB (Neo) ( maximum 3 MCP-Vector/Grapoh DB calls ), have AWS Bedrock generate answer. 

**--------------------------------------------------------------------------------------------------------------------**


*1.1 Start and ingest VECTOR  database* 

docker run --name ai-agent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

Add document to Qdrant  ( create Chunks, Embeddings ):

uv run python -m ingest.ingestVectorDB

[ Check content via dashboard: http://localhost:6333/dashboard#/collections ]

*1.2 Start and ingest GRAPH database*

docker run \
    -p 7474:7474 -p 7687:7687 \
    -v /c/Daten/IT/DEV/Python/RAG_Research/data:/var/lib/neo4j/import \
    --name neo4j \
    -e NEO4J_PLUGINS='["apoc"]' \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

UI: http://localhost:7474/browser/

ingest graph-native document 'neo4j_semiconductor_graph' :

uv run python -m rag.ingest.ingestGraphDB


*2. Start MCP server*

uv run python -m mcpserver


*3. Execute RAG*
run ./scripts/start.sh  (   for AWS Bedrock usage set AWS_REGION & AWS_PROFILE
                            for Claude usage set ANTHROPIC_API_KEY )

prerequisite: manually set same port number in client.py @  "url": "http://localhost:PORTNUMBER/mcp"

uv run python -m mainAwsBedrock

Result:
"""
TODO
"""

**--------------------------------------------------------------------------------------------------------------------**

USE CASE 2 :  via MCP query a VectorDB (Qdrant) and a GraphDB (Neo) ( maximum 3 MCP-Vector/Grapoh DB calls ),direct via Anthropic  

**--------------------------------------------------------------------------------------------------------------------**


*1. Start VectorDB and GraphDB *
see above
*2.* 

*2. Start MCP server*

uv run python -m rag.mcpapi.mcpserver


*3. Retrieve AMD-related subset of ingested content*

run ./scripts/start.sh  (   for AWS Bedrock usage set AWS_REGION & AWS_PROFILE
                            for Claude usage set ANTHROPIC_API_KEY )
uv run python -m mainGraphDB


Result:
"""
Hier sind die Informationen, die ich zu **AMD** in der Graphdatenbank gefunden habe:

🔴 AMD (Advanced Micro Devices)

- 🏭 **Produkte:** AMD stellt **CPUs & KI-Beschleuniger** (AI Accelerators) her.
- 🌍 **Hauptsitz:** Das Unternehmen hat seinen Sitz in den **USA**.

---

AMD ist ein bedeutender Halbleiterhersteller, der vor allem für seine Prozessoren (CPUs) und zunehmend auch für KI-Hardware bekannt ist. Möchtest du mehr über bestimmte Produkte oder Zusammenhänge von AMD erfahren?
"""


**--------------------------------------------------------------------------------------------------------------------**

**--------------------------------------------------------------------------------------------------------------------**

**--------------------------------------------------------------------------------------------------------------------**


BACKGROUND

Einsatz von Graph DBs mittlerweile ein sehr verbreiteter und stark wachsender Trend, der oft als GraphRAG (Graph Retrieval-Augmented Generation) bezeichnet wird.

Während klassisches RAG hauptsächlich auf Vektordatenbanken (Vektorsuche über Text-Embeddings) setzt, 

lösen Graphendatenbanken (wie Neo4j, AWS Neptune oder ArangoDB) entscheidende Schwachstellen dieser Methode.

Warum werden Graphendatenbanken in RAG-Szenarien genutzt?

• Verknüpfung von Informationen: Vektorsuche findet hervorragend ähnliche Textabschnitte, ist aber blind für explizite Beziehungen. Graphen verknüpfen Entitäten (z. B. Person A -> arbeitet bei -> Firma B -> stellt her -> Produkt C).
• Multi-Hop-Anfragen: Fragen, die das Zusammenführen von Informationen über mehrere Ecken erfordern (z. B. „Welche CEOs von Tech-Firmen haben an derselben Universität studiert wie Gründer X?“), sind für reine Vektordatenbanken extrem schwer, für Graphen jedoch Standard.
• Ergebnis-Kontext und Explainability: Graphen bieten eine klare Struktur. Man kann der KI genau vorgeben, welche Pfade sie im Graphen ablaufen soll. Das reduziert Halluzinationen und macht die Antwort nachvollziehbar.

Wie sieht die Kombination in der Praxis aus?

In modernen Enterprise-Architekturen nutzt man meist einen hybriden Ansatz:
1. Vektorsuche (Unstrukturierter Text): Für das Finden von semantisch relevanten Dokumenten oder Abschnitten.
2. Graphsuche (Strukturierte Beziehungen): Für das Abfragen von Fakten, Hierarchien und Netzwerken.
3. Kombination: Das LLM erhält sowohl die Textfragmente als auch die extrahierten Graphen-Zusammenhänge (oft als Wissensgraph / Knowledge Graph), um die finale Antwort zu generieren. Frameworks wie LangChain, LlamaIndex oder Microsofts eigenes GraphRAG-Repository bieten dafür standardisierte Implementierungen.


**--------------------------------------------------------------------------------------------------------------------**

**--------------------------------------------------------------------------------------------------------------------**

**--------------------------------------------------------------------------------------------------------------------**

SANDBOX

*ingest persons.csv document into Neo graph db*

copy from local PC into Docker container :

docker cp persons.csv neo4j:/var/lib/neo4j/import/persons.csv

import csv via query in neo ui ( browser ):

LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS row

MERGE (p:Person {name: row.person_name})

MERGE (c:Company {name: row.company_name, industry: row.company_industry})

MERGE (p)-[:WORKS_AS {role: row.person_role}]->(c)

*get schema
CALL apoc.meta.data()





*delete whole content
MATCH (n) DETACH DELETE n )

*retrieval

uv run python -m rag.retrieval.graphDBPersonRetrieval