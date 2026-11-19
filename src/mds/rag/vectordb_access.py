from fastembed import TextEmbedding, SparseTextEmbedding
from mds.rag.embeddings import create_embedding_model, BM25_MODEL_NAME,RERANKER_MODEL_NAME,SEMANTIC_SEARCH_EMBEDDING_MODEL
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Any


QDRANT_URL = "http://localhost:6333"



def _search(client, limit , collection, query, query_embedding, documents) -> dict:
    try:
        print(f"_search - params: limit: {limit}, collection: {collection}, query: {query}, documents: {documents}")
        result = client.query_points(
            collection_name=collection,
            query=query_embedding.tolist(),
            limit=limit,
            with_payload=True,
        )
        for point in result.points:
            documents.append(
                {
                    "score": point.score,
                    "text": point.payload["text"],
                    "source": point.payload["source"],
                }
            )
    except Exception as e:
        print(f"qdrant_client exception: {e}")

    return rerank(documents, query)

def _search_hybrid(client, limit , collection, query, semantic_search_embedding_model, query_embedding, documents) -> dict:
    """
        apply a semantic and a keyword-based search nd rerank the result
    """
    try:
        print(f"_search_hybrid - params: limit: {limit}, collection: {collection}, query: {query}, documents: {documents}")
        dense_model = TextEmbedding(model_name=semantic_search_embedding_model)
        # Query in beide Vektorformate umwandeln
        # 1. Erzeugt die Liste aus 384 Kommazahlen für die Sinnsuche
        query_dense = list(dense_model.embed([query]))[0].tolist()

        sparse_model = SparseTextEmbedding(
            model_name=BM25_MODEL_NAME
        )
        # 2. Erzeugt das Objekt mit Wort-Positionen und BM25-Gewichten für die Keyword-Suche
        query_sparse = list(sparse_model.embed([query]))[0]

        results = client.query_points(
            collection_name=collection,
            prefetch=[
                # 1. Semantische Suche
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=5,#max number of point to return
                ),
                # 2. Keyword-Suche (BM25)
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist(),#
                    ),
                    using="sparse",
                    limit=5,#max number of point to return
                ),
            ],
            # Beide Ergebnislisten mit Reciprocal Rank Fusion zusammenführen
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        print(f"_search_hybrid  documents: {documents}")
        for hit in results.points:
            print(f"Score: {hit.score:.4f} | Text: {hit.payload['text']}")
    except Exception as e:
        print(f"Exception _search_hybrid: {e}")

    return results.points


def rerank(retrieved_docs : list[str], query:str) -> list[dict]:
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    # 1. RICHTIG: Nur reine Text-Paare [Query, Text] für den Reranker bauen
    text_paare = [[query, item["text"]] for item in retrieved_docs]

    # 2. Neue Reranker-Scores berechnen
    rerank_scores = reranker.predict(text_paare)

    # 3. Neue Scores den ursprünglichen Objekten (inkl. 'source') wieder zuordnen
    for item, neuer_score in zip(retrieved_docs, rerank_scores):
        item["rerank_score"] = float(neuer_score)

    # 4. Nach dem neuen Score absteigend sortieren
    sortierte_treffer = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

    #for t in sortierte_treffer:
    #    print(f"Score: {t['rerank_score']:.4f} | Quelle: {t['source']} | Text: {t['text']}")

    return sortierte_treffer


def _create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
    )

def qdrant_vector_search(
    query: str,
    limit: int = 3,
    collection_name: str = None,
    semantic_search_embedding_model :str = SEMANTIC_SEARCH_EMBEDDING_MODEL,
    hybrid : bool = False
) -> dict[Any, Any] | None:

    documents= []
    try:
        print(f"qdrant_vector_search 1 - query: {query}, hybrid: {hybrid}, collection: {collection_name}")
        e_m = create_embedding_model(semantic_search_embedding_model)
        query_embedding =None
        if(type(e_m) == SentenceTransformer):
            query_embedding  = e_m.encode(query, prompt_name="query")
        elif(type(e_m) == TextEmbedding):
            query_embedding = next(
                e_m.embed([query])
            )
        else:
            print("Query embedding failed!")
            return

        client = _create_qdrant_client()
        info = client.get_collection(collection_name=collection_name)
        print(info.config.params.vectors)

        if hybrid == False:
            documents=_search(client, limit, collection_name, query,  query_embedding, documents)
        else:
            documents=_search_hybrid(client, limit, collection_name, query, semantic_search_embedding_model, query_embedding, documents)

    except Exception as e:
        print(f"qdrant_vector_search Exception: {e}")

    return documents