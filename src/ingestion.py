from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config import DATA_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

def load_documents() -> list[Document]:
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store(documents: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )

    return vector_store

def run_ingestion_pipeline() -> Chroma:
    try:
        documents = load_documents()
        split_docs = split_documents(documents)
        vector_store = create_vector_store(split_docs)
        return vector_store
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found. Please ensure the directory exists and contains PDF files.") from e
    except Exception as e:
        raise RuntimeError("An error occurred during the ingestion pipeline.") from e

