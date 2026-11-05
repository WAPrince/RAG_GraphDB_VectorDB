from qdrant_client import QdrantClient,models
from retrieval.vectorEmbeddings import create_embedding_model

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge"


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
    )

def search_documents(
    query: str,
    limit: int = 3,
) -> list[dict]:
    embedding_model = create_embedding_model()

    query_embedding = next(
        embedding_model.embed([query])
    )

    client = create_qdrant_client()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True,
    )

    documents = []

    for point in result.points:
        documents.append(
            {
                "score": point.score,
                "text": point.payload["text"],
                "source": point.payload["source"],
            }
        )

    return documents