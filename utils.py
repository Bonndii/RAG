import hashlib

from langchain_core.documents import Document

def make_chunk_id(doc: Document) -> str:
    raw = (
        f"{doc.metadata.get('source')}|"
        f"{doc.metadata.get('page')}|"
        f"{doc.page_content}"
    )

    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[source={doc.metadata.get('source')}, page={doc.metadata.get('page')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )