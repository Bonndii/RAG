from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )