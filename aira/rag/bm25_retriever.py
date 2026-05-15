# bm25_retriever.py

from rank_bm25 import BM25Okapi
from langchain.schema import Document
from typing import List
from loguru import logger
import re


class BM25Retriever:
    """
    Keyword-based(TF-IDF) retriever using BM25Okapi algorithm.
    Built from the same document corpus as the FAISS index.
    """

    def __init__(self, documents: List[Document], k: int = 20):
        self.documents = documents
        self.k = k
        self.bm25 = self._build_index(documents)
        logger.info(f"BM25 index built — {len(documents)} documents, fetch_k={k}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + lowercase tokenizer.
        Removes punctuation, splits on whitespace.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def _build_index(self, documents: List[Document]) -> BM25Okapi:
        logger.info("Building BM25 index...")
        tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        return BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> List[Document]:
        """
        Returns top-k documents ranked by BM25 score.
        """
        logger.info(f"BM25 retrieving top-{self.k} for: '{query}'")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each doc with its score, sort descending
        scored_docs = sorted(
            zip(scores, self.documents),
            key=lambda x: x[0],
            reverse=True
        )

        top_docs = [doc for score, doc in scored_docs[:self.k] if score > 0]

        logger.info(f"BM25 returned {len(top_docs)} documents with score > 0")

        for idx, (score, doc) in enumerate(scored_docs[:self.k]):
            if score > 0:
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:80].replace("\n", " ")
                logger.debug(
                    f"[BM25 {idx}] score={score:.4f} "
                    f"source={source} preview={preview}"
                )

        return top_docs