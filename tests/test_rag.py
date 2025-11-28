import pytest
from unittest.mock import patch
from langchain_core.documents import Document


def test_get_rag_chain_fails_without_vector_store() -> None:
    import src.rag

    original_vector_store_dir = src.rag.VECTOR_STORE_DIR
    src.rag.VECTOR_STORE_DIR = "./non_existent_directory"   

    try:
        with pytest.raises(FileNotFoundError, match="Vector store"):
            src.rag.get_rag_chain()
    finally:
        src.rag.VECTOR_STORE_DIR = original_vector_store_dir


def test_format_docs(sample_documents: list[Document]) -> None:
    def format_docs(docs: list[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    result = format_docs(sample_documents)

    assert "doc1.pdf" in result
    assert "Page: 1" in result

def test_format_docs_handles_missing_metadata() -> None:
    docs = [Document(page_content="Without metadata", metadata={})]

    def format_docs(docs: list[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    result = format_docs(docs)

    assert "unknown" in result
    assert "Without metadata" in result