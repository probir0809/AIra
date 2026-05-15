# rag_tool.py

from loguru import logger
from typing import List
from langchain.schema import Document
from aira.rag.kb_metadata import KBMetadata


class RAGTool:
    """
    Wraps the full retrieval pipeline as a callable tool.
    Description is dynamically built from kb_metadata.json
    and uploads_metadata.json.
    """

    name = "document_search"

    def __init__(self, retriever):
        self.retriever = retriever
        self.description = self._build_description()
        logger.info(f"RAGTool description: {self.description}")

    def _build_description(self) -> str:
        """Reads both metadata files and builds the agent-facing description."""
        try:
            metadata = KBMetadata()
            return metadata.build_rag_tool_description()
        except Exception as e:
            logger.warning(f"Could not build description: {e}")
            return (
                "Searches the private document knowledge base. "
                "Use this BEFORE web_search for domain-specific questions."
            )

    def run(self, query: str) -> str:
        logger.info(f"RAGTool query: '{query}'")
        try:
            documents = self.retriever.retrieve(query)
            if not documents:
                return "No relevant documents found in the knowledge base."
            return self._format_documents(documents)
        except Exception as e:
            logger.error(f"RAGTool error: {e}")
            return f"Document search error: {e}"

    def _format_documents(self, documents: List[Document]) -> str:
        lines = [f"Retrieved {len(documents)} relevant document chunks:\n"]
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown").split("/")[-1]
            lines.append(f"[Chunk {i}] Source: {source}")
            lines.append(doc.page_content.strip())
            lines.append("")
        return "\n".join(lines)