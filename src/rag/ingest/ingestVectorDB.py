
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from qdrant_client import models
from retrieval.vectorEmbeddings import create_embedding_model
from retrieval.vectorDbRetrieval import COLLECTION_NAME
from retrieval.vectorStore import create_qdrant_client



DATA_FILE = Path("/data/input/vectordb_source_v2.txt")

def _load_document(file :str ) -> str:
    return file.read_text(
        encoding="utf-8",
    )

# fixed chunk size => semantic search will affected ( completeness accuracy consistency costs )+
def _split_into_chunks(
    text: str,
    chunk_size: int = 300,
) -> list[str]:
    words = text.split()

    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks

def _ingest(file : str ) -> None:
    try:
        text = _load_document(file)
        start = datetime.now()
        chunks = _split_into_chunks(text)
        print(f"Document contains {len(text)} characters")
        print(f"Created {len(chunks)} chunks")
        embedding_model = create_embedding_model()
        print(f"embeddding_model")
        embeddings = list(
            embedding_model.embed(chunks)
        )
        print(f"embeddings")
        vector_size = len(embeddings[0])
        print(f"Vector size {vector_size}")
        # load embeddings to Qdrant vector database in docker container
        client = create_qdrant_client()
        collections = client.get_collections()
        collection_exists = any(
            collection.name == COLLECTION_NAME
            for collection in collections.collections
        )
        if not collection_exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                #Hierarchical navigable small world parameters !
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )
        #Beim HNSW-Algorithmus (Hierarchical Navigable Small Worlds) für die Vektorsuche steuern die beiden Parameter M und ef_construction, wie die mehrschichtige Graphstruktur während der Indexierung aufgebaut wird. Sie bestimmen die Balance zwischen Indexierungsgeschwindigkeit, Speicherbedarf und der späteren Suchgenauigkeit (Recall).
        #  1. M (Maximum Connections)
        #M definiert die maximale Anzahl von Verbindungen (Kanten), die ein Vektorknoten zu anderen Knoten im Graphen aufbauen darf. Auf der untersten Ebene (Layer 0) wird dieser Wert oft auf 2 × M verdoppelt, um die Stabilität bei dichten Graphen zu erhöhen.
        #• Typischer Wertebereich: 8 bis 64.
        #• Niedriger Wert (z. B. 8–16): Weniger Speicherverbrauch (RAM) und schnellere Indexierung, aber die Suchqualität (Recall) kann bei komplexen Daten sinken.
        #• Hoher Wert (z. B. 32–64): Erzeugt ein dichteres Netzwerk mit mehr Pfaden. Das führt zu einer höheren Suchgenauigkeit (höherer Recall), erhöht jedoch den Speicherbedarf des Index linear und verlangsamt den Indexaufbau.
        #
        #  2. ef_construction (Evaluation Factor Exploration)
        #ef_construction bestimmt die Größe der dynamischen Kandidatenliste, die während des Indexaufbaus durchsucht wird, um die besten Nachbarn für einen neuen Vektor zu finden. Es steuert sozusagen, wie „gründlich“ oder „tief“ der Algorithmus beim Einfügen jedes neuen Elements nach optimalen Verbindungen sucht.
        #• Typischer Wertebereich: 100 bis 500.
        #• Wie es funktioniert: Beim Einfügen prüft der Algorithmus nicht nur die direkt nächsten Nachbarn, sondern hält einen Pool von ef_construction-Kandidaten offen. Ein höherer Wert führt dazu, dass die Verbindungen im Graphen mathematisch präziser gewählt werden (bessere Graphqualität).
        #• Auswirkung: Ein sehr hoher Wert verlängert die Zeit für den Indexaufbau (Build Time) massiv, hat aber keinen Einfluss auf den Speicherbedarf des fertigen Index (da die finale Anzahl der Kanten fest durch M begrenzt bleibt). Er sorgt lediglich dafür, dass die durch M definierten Kanäle von besserer Qualität sind.

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            print(f"chunk {chunk.index}")
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk,
                        "source": str(file),
                    },
                )
            )
        print(f"Upsert")
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        end = datetime.now()
        duration = end - start
        print(f"Einlesedauer in Sekunden: {duration.total_seconds()}")
        print(f"End")
    except Exception as e:
        print(f"Exception ${e.str}")

if __name__ == "__main__":
    _ingest(DATA_FILE)