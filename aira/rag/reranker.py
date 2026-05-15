# reranker.py

from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import List
from loguru import logger
from aira.core.config import RERANKER_MODEL_NAME, RERANKER_TOP_K


class CrossEncoderReranker:
    """
    Re-ranks retrieved documents using a CrossEncoder model.
    
    Flow: FAISS top-N (coarse) → CrossEncoder scores → top-k (fine)
    """

    def __init__(self):
        logger.info(f"Loading CrossEncoder reranker: {RERANKER_MODEL_NAME}")
        self.model = CrossEncoder(RERANKER_MODEL_NAME)
        self.top_k = RERANKER_TOP_K
        logger.info("CrossEncoder loaded successfully")

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Score each (query, doc) pair and return top-k by score.
        """
        if not documents:
            logger.warning("Reranker received empty document list")
            return []

        logger.info(f"Reranking {len(documents)} documents → top {self.top_k}")

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # CrossEncoder scores all pairs in one forward pass
        scores = self.model.predict(pairs)

        # Attach scores and sort descending
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        # Log scores for debugging
        for i, (score, doc) in enumerate(scored_docs[:self.top_k]):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            logger.debug(f"[RERANK {i}] score={score:.4f} source={source} preview={preview}")

        top_docs = [doc for _, doc in scored_docs[:self.top_k]]
        logger.info(f"Reranking complete — returning {len(top_docs)} documents")

        return top_docs