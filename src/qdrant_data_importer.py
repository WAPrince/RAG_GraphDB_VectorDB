from pathlib import Path
from uuid import uuid4
from datetime import datetime

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from mds.rag.vector_store import _create_qdrant_client

from mds.rag.embeddings import create_embedding_model, BAAI_MODEL_NAME,MINILM_MODEL_NAME,HARRIER_MODEL_NAME,QWEN_MODEL_NAME,BM25_MODEL_NAME,RERANKER_MODEL_NAME,SEMANTIC_SEARCH_EMBEDDING_MODEL
import pathlib

# Paths are valid if program started from src folder
SEMICONDUCTOR_PATH = Path("../data/input/semiconductor/vector/")


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start
        print(f"Ingestion took {duration.total_seconds()} seconds")
        return result
    return wrapper

def load_document(file :str ) -> str:
    return file.read_text(
        encoding="utf-8",
    )

# fixed chunk size => semantic search will affected ( completeness accuracy consistency costs )+
def _split_into_chunks(
    file: str,
    chunk_size: int = 300,
) -> list[str]:
    text = load_document(file)
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    print(f"Document contains {len(text)} characters")
    print(f"Created {len(chunks)} chunks")
    return chunks

def _get_embeddings(semantic_search_embedding_model_id:str, chunks) -> list[dict]:
    data_embeddings = None
    embedding_model = create_embedding_model(semantic_search_embedding_model_id)
    if (type(embedding_model) == SentenceTransformer):
        print(f"_get_embeddings {embedding_model}")
        data_embeddings = embedding_model.encode(chunks, prompt_name="query")
    elif (type(embedding_model) == TextEmbedding):
        print(f"_get_embeddings {embedding_model}")
        data_embeddings = list(
            embedding_model.embed(chunks)
        )
    else:
        print("Query embedding failed!")
        return
    return data_embeddings


def _path_to_collection(path:str):
    if path==SEMICONDUCTOR_PATH:
        return "semiconductor"
    else:
        print(f"path {path} unknown!")

def _collection(client, vector_size, path):
    collections = client.get_collections()
    collection_exists = any(
        collection.name == _path_to_collection(path)
        for collection in collections.collections
    )
    if not collection_exists:
        client.create_collection(
            collection_name=_path_to_collection(path),
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
    return


def _collection_dense_sparse(client, path):
    """
        create vectors for hybrid search: semantic search  => dense & keyword search => sparse
    """

    collections = client.get_collections()
    collection__dense_saprse_exists = any(
        collection.name == _path_to_collection(path)+"__dense_sparse"
        for collection in collections.collections
    )
    if not collection__dense_saprse_exists:

        client.create_collection(
            collection_name=_path_to_collection(path)+"__dense_sparse",
            vectors_config={
                "dense": models.VectorParams(
                    size=384,  # Passend zu deinem Dense-Modell (z.B. all-MiniLM-L6-v2)
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    # Wichtig für BM25-Gewichtung
                )
            }
        )
        print(f"Collection {_path_to_collection(path)+"__dense_sparse"} created.")
    else:
        print(f"Collection already exists.")



def _upsert(file, client, chunks, data_embeddings, path):
    current_count = client.count(collection_name=_path_to_collection(path)).count
    points = []
    idx:int=0
    for chunk, embedding in zip(chunks, data_embeddings):
        print(f"chunk {chunk.index}")
        points.append(
            models.PointStruct(
                id=current_count + idx,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": str(file),
                },
            )
        )
        idx=idx+1
    client.upsert(
        collection_name=_path_to_collection(path),
        points=points,
    )

def _upsert_dense_space(client, chunks: list, data_embeddings, path):

    current_sparse_dense_count = client.count(collection_name=_path_to_collection(path)+"__dense_sparse").count
    dense_model = TextEmbedding(model_name=SEMANTIC_SEARCH_EMBEDDING_MODEL)
    sparse_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

    # Embeddings generieren
    dense_embeddings = list(dense_model.embed(chunks))
    sparse_embeddings = list(sparse_model.embed(chunks))

    points = []
    for idx, (doc, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
        points.append(
            models.PointStruct(
                id=current_sparse_dense_count + idx,
                payload={"text": doc},
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": models.Document(
                        text=doc,
                        model=BM25_MODEL_NAME,#"Qdrant/bm25",
                        options={"language": "german"} # Optional für deutsche Stoppwörter/Stemming
                    )
                }
            )
        )

    client.upsert(collection_name=_path_to_collection(path)+"__dense_sparse", points=points)



@measure_time
def ingest(file : str, semantic_search_embedding_model_id : str, path:str) -> None:

    try:
        chunks = _split_into_chunks(file=file)

        data_embeddings =  _get_embeddings(semantic_search_embedding_model_id, chunks)

        client = _create_qdrant_client()

        #_collection(client, vector_size=len(data_embeddings[0]), path=path)
        _collection_dense_sparse(client,path)

        #_upsert(file,client,chunks, data_embeddings, path)
        _upsert_dense_space(client, chunks, data_embeddings, path)

    except Exception as e:
        print(f"Exception ${e}")


if __name__ == "__main__":

    input_folder = pathlib.Path(SEMICONDUCTOR_PATH)
    # .iterdir() produces a generator Which you can wrap in a list() constructor to materialize
    for f in list(input_folder.iterdir()):
        if f.name.endswith(".txt")==False :
            continue

        print(f"Ingest file {f}")

        #configure SEMANTIC_SEARCH_EMBEDDING_MODEL in .env
        ingest(f, SEMANTIC_SEARCH_EMBEDDING_MODEL, SEMICONDUCTOR_PATH)


        #ingest(f, BAAI_MODEL_NAME , SEMICONDUCTOR_PATH)
        #ingest(f, HARRIER_MODEL_NAME , SEMICONDUCTOR_PATH)


        #vector size 1024
        #ingest(FILE, HARRIER_MODEL_NAME, SEMICONDUCTOR_PATH)