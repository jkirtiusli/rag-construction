import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from pytest_mock import MockerFixture

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(page_content="This is a test document.", metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content="Another test document content.", metadata={"source": "doc2.pdf", "page": 2}),
    ]   

@pytest.fixture
def mock_openai_embeddings(mocker: MockerFixture) -> MagicMock:
    mock = mocker.patch("langchain_openai.OpenAIEmbeddings")
    mocker.return_value_embed_documents.return_value = [[0.1] * 1536]  
    mock.return_value.embed_query.return_value = [0.1] * 1536
    return mock