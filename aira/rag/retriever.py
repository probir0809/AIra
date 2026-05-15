# retriever.py

from typing import List
from langchain.schema import Document
from loguru import logger
from aira.core.config import RETRIEVER_TOP_K


class VectorRetriever:
    """
    FAISS retriever — raw dense search.
    No reranker here — reranking is handled upstream
    by the pipeline in dependencies.py after hybrid fusion.
    """

    def __init__(self, vectorstore, k: int = RETRIEVER_TOP_K):
        self.vectorstore = vectorstore
        self.k = k

        self._retriever = self.vectorstore.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )

        logger.info(f"VectorRetriever initialized — fetch_k={self.k}")

    def retrieve(self, query: str) -> List[Document]:
        logger.info(f"FAISS retrieving top-{self.k} for: '{query}'")

        documents = self._retriever.get_relevant_documents(query)
        logger.info(f"FAISS returned {len(documents)} documents")

        for idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:120].replace("\n", " ")
            logger.debug(
                f"[FAISS {idx}] source={source} "
                f"chars={len(doc.page_content)} preview={preview}"
            )

        return documents