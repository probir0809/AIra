# summarizer.py

from langchain.schema import Document
from typing import List
from loguru import logger
import re


class MapReduceSummarizer:
    """
    Summarizes a document using map-reduce strategy.

    Map phase:
      Each chunk → LLM → 1-line summary

    Reduce phase:
      All chunk summaries combined → LLM → final 1-line description

    Works with any document size and any small LLM.
    Never exceeds context window because each step
    only processes one chunk at a time.

    Fallback:
      If the LLM returns an empty string after cleaning,
      the filename is used as the description so metadata
      is never saved as an empty string.
    """

    def __init__(self, llm):
        self.llm = llm
        logger.info("MapReduceSummarizer initialized")

    def _clean(self, text: str) -> str:
        """Remove LLM artifacts like <think> tags and special tokens."""
        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove orphaned closing </think> tags (thinking was cut mid-way)
        text = re.sub(r"</think>", "", text)
        # Remove orphaned opening <think> tags
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        for token in ["<|assistant|>", "<|im_end|>", "<|system|>", "<|user|>"]:
            text = text.replace(token, "")
        return text.strip()

    def _filename_fallback(self, filename: str) -> str:
        """
        Derives a readable description from the filename.
        Used when the LLM returns an empty string.
        e.g. 'my_research_paper.pdf' → 'my research paper'
        """
        name = filename.replace(".pdf", "")
        name = name.replace("_", " ").replace("-", " ")
        # remove version suffixes like v2, v3
        name = re.sub(r"\bv\d+\b", "", name).strip()
        return name.lower()

    def _map_chunk(self, chunk: Document) -> str:
        prompt = (
            "<|system|>\n"
            "You are a summarizer. Summarize the given text in 4-6 words ONLY. "
            "Example outputs: 'scientific paper writing guide', 'machine learning basics', "
            "'research methodology and structure'. "
            "No sentences. No punctuation. No explanation. Just 4-6 words.\n"
            "<|im_end|>\n"
            "<|user|>\n"
            f"Text:\n{chunk.page_content[:600]}\n"
            "<|im_end|>\n"
            "<|assistant|>\n"
        )
        result = self.llm(prompt)
        cleaned = self._clean(result)
        if not cleaned:
            logger.warning(f"Map chunk returned empty after cleaning — skipping chunk")
        return cleaned

    def _reduce(self, chunk_summaries: List[str], filename: str) -> str:
        # Filter out any empty chunk summaries before reduce
        valid_summaries = [s for s in chunk_summaries if s]

        if not valid_summaries:
            logger.warning(f"No valid chunk summaries to reduce for '{filename}'")
            return ""

        combined = "\n".join(f"- {s}" for s in valid_summaries)
        prompt = (
            "<|system|>\n"
            "You are a summarizer. Given multiple short summaries, write ONE final "
            "description of 4-8 words describing what the document is about. "
            "Example outputs: 'guide for writing research papers', "
            "'scientific article structure and formatting'. "
            "No sentences. No punctuation. No explanation. Just 4-8 words.\n"
            "<|im_end|>\n"
            "<|user|>\n"
            f"Document: {filename}\n"
            f"Summaries:\n{combined}\n"
            "<|im_end|>\n"
            "<|assistant|>\n"
        )
        result = self.llm(prompt)
        return self._clean(result)

    def summarize(self, documents: List[Document], filename: str) -> str:
        """
        Full map-reduce pipeline for a single document.

        documents — all chunks from one PDF file
        filename  — used as context hint in reduce phase
        returns   — one final description string, never empty
        """
        logger.info(
            f"Summarizing '{filename}' — "
            f"{len(documents)} chunks via map-reduce"
        )

        if not documents:
            fallback = self._filename_fallback(filename)
            logger.warning(f"No documents to summarize — using fallback: '{fallback}'")
            return fallback

        # MAP — summarize each chunk independently
        chunk_summaries = []
        for i, chunk in enumerate(documents):
            summary = self._map_chunk(chunk)
            chunk_summaries.append(summary)
            logger.debug(f"[MAP {i+1}/{len(documents)}] {summary}")

        # REDUCE — combine all summaries into one final line
        final = self._reduce(chunk_summaries, filename)

        # Fallback — if LLM returned nothing useful after cleaning
        if not final:
            final = self._filename_fallback(filename)
            logger.warning(
                f"Summarizer returned empty after reduce — "
                f"using filename fallback: '{final}'"
            )

        logger.info(f"Final summary for '{filename}': {final}")
        return final