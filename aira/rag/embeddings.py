from langchain_community.embeddings import HuggingFaceEmbeddings
from aira.core.config import EMBEDDING_MODEL_NAME
from loguru import logger

class EmbeddingModel:
    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self._model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

    def get(self):
        return self._model
