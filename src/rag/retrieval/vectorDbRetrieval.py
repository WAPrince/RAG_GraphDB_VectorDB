from qdrant_client import QdrantClient
from retrieval.vectorEmbeddings import create_embedding_model

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge"

def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
    )

# semantic search
def search_documents_in_vectordb(query:str, limit: int = 3) -> list[dict]:

    print("search_documents_in_vectordb")
    embedding_model = create_embedding_model()
    query_embedding = next(
        embedding_model.embed([query])
    )
    client = create_qdrant_client()
    # search similar vectors in qdrant
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True,
    )
    documents = []
    # extract search results
    for point in result.points:
        documents.append(
            {
                "score": point.score,
                "text": point.payload["text"],
                "source": point.payload["source"]
            }
        )
    print(f"search_documents_in_vectordb documents : {documents}")

    return documents
