import pytest
from unittest.mock import patch

def test_open_api_key_missing() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set."):
            import importlib
            import src.config
            importlib.reload(src.config)

def test_config_constanst_exists() -> None:
      from src.config import (
          DATA_DIR,
          VECTOR_STORE_DIR,
          CHUNK_SIZE,
          CHUNK_OVERLAP,
          EMBEDDING_MODEL_NAME,
          LLM_MODEL_NAME,
      )
    
      assert isinstance(DATA_DIR, str)
      assert isinstance(VECTOR_STORE_DIR, str)
      assert isinstance(CHUNK_SIZE, int)
      assert isinstance(CHUNK_OVERLAP, int)
      assert CHUNK_SIZE > CHUNK_OVERLAP
