from unittest.mock import patch
from langchain_core.documents import Document
from src.ingestion import split_documents

def test_split_documents(sample_documents: list[Document]) -> None:
    with patch("src.ingestion.CHUNK_SIZE", 20):
        with patch("src.ingestion.CHUNK_OVERLAP", 5):
            result = split_documents(sample_documents)

    assert isinstance(result, list)
    assert all(isinstance(doc, Document) for doc in result)

def test_split_documents_preserves_metadata(sample_documents: list[Document]) -> None:
    result = split_documents(sample_documents)

    for doc in result:
        assert "source" in doc.metadata
        assert "page" in doc.metadata

def test_split_documents_empty_list() -> None:
    result = split_documents([])

    assert result == []