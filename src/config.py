import os
from dotenv import load_dotenv  
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
DATA_DIR: str = "./data/raw_pdfs"
VECTOR_STORE_DIR: str = "./data/vector_store"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
LLM_MODEL_NAME: str = "gpt-4o"