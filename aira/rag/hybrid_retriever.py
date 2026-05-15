# hybrid_retriever.py

from langchain.schema import Document
from typing import List, Dict
from loguru import logger
from aira.rag.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Fuses FAISS (dense) + BM25 (sparse) results using
    Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = 1/(k + rank_faiss) + 1/(k + rank_bm25)
    k=60 is standard — smooths out high-rank dominance.
    Documents appearing in both lists get boosted.
    Documents in only one list still contribute.
    """

    def __init__(
        self,
        faiss_retriever,       # VectorRetriever (without reranker — raw FAISS)
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        top_n: int = 30        # candidates passed to reranker
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.top_n = top_n
        logger.info(f"HybridRetriever initialized — RRF k={rrf_k}, top_n={top_n}")

    def _get_doc_id(self, doc: Document) -> str:
        """
        Unique identifier for a document chunk.
        Uses source + first 80 chars of content.
        """
        source = doc.metadata.get("source", "")
        return f"{source}::{doc.page_content[:80]}"

    def _rrf_score(
        self,
        faiss_docs: List[Document],
        bm25_docs: List[Document]
    ) -> List[Document]:
        """
        Applies RRF to merge two ranked lists.
        Returns deduplicated list sorted by combined RRF score.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Score from FAISS ranking
        for rank, doc in enumerate(faiss_docs):
            doc_id = self._get_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[doc_id] = doc

        # Score from BM25 ranking (added on top)
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._get_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[doc_id] = doc

        # Sort by combined score descending
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        for idx, doc_id in enumerate(sorted_ids[:self.top_n]):
            logger.debug(
                f"[RRF {idx}] score={scores[doc_id]:.5f} "
                f"preview={doc_id[doc_id.find('::')+2:][:60]}"
            )

        return [doc_map[doc_id] for doc_id in sorted_ids[:self.top_n]]

    def retrieve(self, query: str) -> List[Document]:
        """
        1. FAISS retrieves top-20 semantic candidates
        2. BM25 retrieves top-20 keyword candidates
        3. RRF merges + deduplicates → top_n candidates
        """
        logger.info(f"HybridRetriever: '{query}'")

        faiss_docs = self.faiss_retriever.retrieve(query)
        logger.info(f"FAISS returned {len(faiss_docs)} docs")

        bm25_docs = self.bm25_retriever.retrieve(query)
        logger.info(f"BM25 returned {len(bm25_docs)} docs")

        fused_docs = self._rrf_score(faiss_docs, bm25_docs)
        logger.info(f"RRF fusion → {len(fused_docs)} unique candidates")

        return fused_docs