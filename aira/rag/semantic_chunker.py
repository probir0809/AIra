# semantic_chunker.py

from langchain.schema import Document
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np
from aira.core.config import (
    EMBEDDING_MODEL_NAME,
    SEMANTIC_CHUNK_MIN_SIZE,
    SEMANTIC_CHUNK_MAX_SIZE,
    SEMANTIC_BREAKPOINT_THRESHOLD,
)


class SemanticChunker:
    """
    Splits documents at meaning boundaries instead of fixed character counts.

    How it works:
    1. Split document into sentences
    2. Embed each sentence using the same embedding model as FAISS
    3. Compute cosine similarity between adjacent sentence embeddings
    4. When similarity drops below threshold → topic has changed → cut here
    5. Merge sentences between cuts into chunks, respecting min/max size
    """

    def __init__(self):
        logger.info(f"Loading embedding model for semantic chunker: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.min_size = SEMANTIC_CHUNK_MIN_SIZE
        self.max_size = SEMANTIC_CHUNK_MAX_SIZE
        self.threshold = SEMANTIC_BREAKPOINT_THRESHOLD
        logger.info(
            f"SemanticChunker ready — "
            f"min={self.min_size} max={self.max_size} "
            f"threshold={self.threshold}"
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Naive but effective sentence splitter.
        Splits on '.', '!', '?' followed by whitespace.
        Filters out empty or very short fragments.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _find_breakpoints(self, sentences: List[str]) -> List[int]:
        """
        Embeds all sentences, computes similarity between neighbors,
        returns indices where similarity drops below threshold (topic shift).
        """
        if len(sentences) <= 1:
            return []

        logger.debug(f"Embedding {len(sentences)} sentences...")
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        breakpoints = []
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
            if sim < self.threshold:
                breakpoints.append(i + 1)  # cut before sentence i+1
                logger.debug(f"Breakpoint at sentence {i+1} — similarity={sim:.4f}")

        if similarities:
            logger.debug(
                f"Similarity stats — "
                f"min={min(similarities):.4f} "
                f"max={max(similarities):.4f} "
                f"avg={np.mean(similarities):.4f} "
                f"breakpoints={len(breakpoints)}"
            )

        return breakpoints

    def _merge_sentences(
        self,
        sentences: List[str],
        breakpoints: List[int],
        metadata: dict
    ) -> List[Document]:
        """
        Groups sentences between breakpoints into chunks.
        Respects min_size and max_size limits.
        """
        chunks = []
        breakpoints_set = set(breakpoints)

        current_sentences = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            # Start new chunk at breakpoint OR if current chunk exceeds max size
            if (i in breakpoints_set or current_length + len(sentence) > self.max_size) \
                    and current_length >= self.min_size:

                chunk_text = " ".join(current_sentences)
                chunks.append(Document(page_content=chunk_text, metadata=metadata.copy()))
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += len(sentence)

        # Don't forget the last chunk
        if current_sentences and current_length >= self.min_size:
            chunk_text = " ".join(current_sentences)
            chunks.append(Document(page_content=chunk_text, metadata=metadata.copy()))

        return chunks

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Main entry point — mirrors the interface of TextChunker.split().
        Drop-in replacement.
        """
        logger.info(f"SemanticChunker: processing {len(documents)} documents")
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            source = doc.metadata.get("source", f"doc_{doc_idx}")
            logger.debug(f"Processing: {source} ({len(doc.page_content)} chars)")

            sentences = self._split_into_sentences(doc.page_content)

            if not sentences:
                logger.warning(f"No sentences extracted from {source} — skipping")
                continue

            breakpoints = self._find_breakpoints(sentences)
            chunks = self._merge_sentences(sentences, breakpoints, doc.metadata)

            logger.debug(f"{source} → {len(sentences)} sentences → {len(chunks)} chunks")
            all_chunks.extend(chunks)

        logger.info(
            f"SemanticChunker complete — "
            f"{len(documents)} docs → {len(all_chunks)} chunks"
        )
        return all_chunks