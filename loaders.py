import pypdf
from langchain_core.documents import Document

def load_pdf_pages(file_path: str, file_name: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)

    docs = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": i,
                    },
                )
            )

    return docs