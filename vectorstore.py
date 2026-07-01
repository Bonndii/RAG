from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from config import QDRANT_URL
from utils import make_chunk_id


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def get_collection_names() -> list[str]:
    client = get_qdrant_client()
    collections = client.get_collections().collections

    return [collection.name for collection in collections]


def create_vectorstore_from_documents(
    documents: list[Document],
    embeddings,
    collection_name: str,
) -> QdrantVectorStore:
    ids = [make_chunk_id(doc) for doc in documents]

    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        url=QDRANT_URL,
        collection_name=collection_name
    )


def load_existing_vectorstore(
    embeddings,
    collection_name: str,
) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name
    )