import os
from dotenv import load_dotenv
import torch
from fastembed import TextEmbedding
from numpy.f2py.auxfuncs import throw_error
from sentence_transformers import SentenceTransformer
from typing_extensions import Any


load_dotenv()

#semantic search ######################################################################################################
# vector size 364
#	• Einsatzgebiet: Der bewährte Spezialist für deutsche und mehrsprachige Dokumente mit Hybrid-Suche.
#	• Spezifikationen: 568M Parameter, 8.192 Tokens Kontext.
#	  s beherrscht gleichzeitig semantische Suche (Dense), klassische Keyword-Suche (Sparse) und Multi-Vector-Retrieval in einem Modell.
BAAI_MODEL_NAME = os.getenv("BAAI_MODEL_NAME") # "BAAI/bge-small-en-v1.5"

# nur dense! separater Schritt für  sparse, z.Bsp mit BM25 !
# vector size 384
MINILM_MODEL_NAME =os.getenv("MINILM_MODEL_NAME") # "sentence-transformers/all-MiniLM-L6-v2"

# vector size 1024
HARRIER_MODEL_NAME = os.getenv("HARRIER_MODEL_NAME") # "microsoft/harrier-oss-v1-0.6b"

QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME") # "Qwen/Qwen3-Embedding-8B"

SEMANTIC_SEARCH_EMBEDDING_MODEL = os.getenv("SEMANTIC_SEARCH_EMBEDDING_MODEL")


#keyword search #######################################################################################################
BM25_MODEL_NAME = os.getenv("BM25_MODEL_NAME") # "Qdrant/bm25"
SPLADE_MODEL_NAME  = os.getenv("SPLADE_MODEL_NAME") # naver/splade-cocondenser-ensembledistil


#hybrid search ########################################################################################################
#Dense-, Sparse- und Multi-Vector-Retrieval
BAAIM3_MODEL_NAME = os.getenv("BAAIM3_MODEL_NAME") # "BAAI/bge-m3"


#reranking ############################################################################################################
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME") #"BAAI/bge-reranker-v2-m3"




def create_embedding_model(embedding_model_id:str) -> Any:

    if (embedding_model_id == BAAI_MODEL_NAME):
        print(f"create_embedding_model {BAAI_MODEL_NAME}")
        return TextEmbedding(
            model_name=BAAI_MODEL_NAME,
        )
    elif (embedding_model_id == MINILM_MODEL_NAME):
        print(f"create_embedding_model {MINILM_MODEL_NAME}")
        return TextEmbedding(
            model_name=MINILM_MODEL_NAME,
        )
    elif (embedding_model_id == HARRIER_MODEL_NAME):
        print(f"create_embedding_model {HARRIER_MODEL_NAME}")
        return  SentenceTransformer(
            HARRIER_MODEL_NAME, #os.getenv("HARRIER_MODEL_NAME"),#"microsoft/harrier-oss-v1-0.6b",
            model_kwargs={"dtype": torch.float32},#{"torch_dtype": "auto", "device_map": "auto"},
        )
    elif (embedding_model_id == QWEN_MODEL_NAME):
        print(f"create_embedding_model {QWEN_MODEL_NAME}")
        return  SentenceTransformer(
            QWEN_MODEL_NAME,#os.getenv("QWEN_MODEL_NAME"),#"Qwen/Qwen3-Embedding-8B",
            model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
        )
    else:
       throw_error(f"Embedding model {embedding_model_id} not supported!")
