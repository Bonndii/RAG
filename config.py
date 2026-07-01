import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "xiaomi/mimo-v2.5-pro")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))