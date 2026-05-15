from langchain_community.vectorstores import FAISS
from typing import List
from langchain.schema import Document
from loguru import logger
import os


class FAISSVectorStore:
    """
    Wrapper around LangChain FAISS vectorstore
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectorstore = None

    def build(self, documents: List[Document]):
        logger.info("Building FAISS index from documents")
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding_model
        )
        return self.vectorstore

    def load(self, path: str):
        logger.info(f"Loading FAISS index from {path}")
        self.vectorstore = FAISS.load_local(
            path,
            self.embedding_model
        )
        return self.vectorstore

    def save(self, path: str):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not built yet")

        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving FAISS index to {path}")
        self.vectorstore.save_local(path)

    def as_retriever(self, k: int = 4):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded or built")

        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
