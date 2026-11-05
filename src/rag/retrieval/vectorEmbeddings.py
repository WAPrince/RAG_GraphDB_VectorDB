from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def create_embedding_model() -> TextEmbedding:
    return TextEmbedding(
        model_name=MODEL_NAME,
    )