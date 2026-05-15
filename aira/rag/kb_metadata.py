# kb_metadata.py

import json
import os
from loguru import logger
from typing import Dict

BASE_METADATA_PATH = "data/kb_metadata.json"
UPLOADS_METADATA_PATH = "data/uploads_metadata.json"


class KBMetadata:
    """
    Manages two metadata JSON files:

    kb_metadata.json      — base knowledge base (build_faiss.py)
                            permanent, never wiped automatically

    uploads_metadata.json — user uploaded documents (api/documents.py)
                            session-scoped, wiped on server shutdown
                            to stay in sync with the in-memory session
                            vectorstore which is also gone at that point

    Each file is a dict: { "filename.pdf": "one line description" }
    """

    def __init__(
        self,
        base_path: str = BASE_METADATA_PATH,
        uploads_path: str = UPLOADS_METADATA_PATH
    ):
        self.base_path = base_path
        self.uploads_path = uploads_path
        os.makedirs("data", exist_ok=True)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load(self, path: str) -> Dict[str, str]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata from {path}: {e}")
            return {}

    def _save(self, path: str, data: Dict[str, str]):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Metadata saved to {path}")

    # ── Base KB (permanent) ─────────────────────────────────────────────────

    def add_base_doc(self, filename: str, description: str):
        """Add or update a base KB document description."""
        data = self._load(self.base_path)
        data[filename] = description
        self._save(self.base_path, data)
        logger.info(f"Base KB metadata updated: {filename} → {description}")

    def get_base_docs(self) -> Dict[str, str]:
        """Returns all base KB document descriptions."""
        return self._load(self.base_path)

    # ── Upload KB (session-scoped) ──────────────────────────────────────────

    def add_upload_doc(self, filename: str, description: str):
        """Add or update a user upload description."""
        data = self._load(self.uploads_path)
        data[filename] = description
        self._save(self.uploads_path, data)
        logger.info(f"Upload metadata updated: {filename} → {description}")

    def get_upload_docs(self) -> Dict[str, str]:
        """Returns all session upload document descriptions."""
        return self._load(self.uploads_path)

    def remove_upload_doc(self, filename: str):
        """
        Remove a single doc from upload metadata.
        Useful for a future 'delete upload' API endpoint.
        """
        data = self._load(self.uploads_path)
        if filename in data:
            del data[filename]
            self._save(self.uploads_path, data)
            logger.info(f"Upload metadata removed: {filename}")

    def clear_upload_docs(self):
        """
        Wipes uploads_metadata.json entirely.
        Called on server shutdown via the lifespan handler in main.py
        so metadata stays in sync with the in-memory session vectorstore
        which is also gone at that point.
        """
        self._save(self.uploads_path, {})
        logger.info("Upload metadata cleared — session ended")

    # ── RAG tool description ────────────────────────────────────────────────

    def build_rag_tool_description(self) -> str:
        """
        Builds the dynamic RAGTool description from both metadata files.
        This is what the agent reads to decide when to use the RAG tool.

        Output example:
        'Searches the private document knowledge base. Use this BEFORE
         web_search for domain-specific questions. We have knowledge base
         related to "guide for writing research papers". We also have
         user-uploaded knowledge base related to "machine learning textbook".'
        """
        base_docs = self.get_base_docs()
        upload_docs = self.get_upload_docs()

        lines = [
            "Searches the private document knowledge base. "
            "Use this BEFORE web_search for domain-specific questions. "
            "Input must be a clear question or search query string."
        ]

        if base_docs:
            descriptions = ", ".join(
                f'"{desc}"' for desc in base_docs.values() if desc
            )
            if descriptions:
                lines.append(
                    f"We have knowledge base related to {descriptions}."
                )

        if upload_docs:
            descriptions = ", ".join(
                f'"{desc}"' for desc in upload_docs.values() if desc
            )
            # Only append sentence if at least one upload has a non-empty description
            if descriptions:
                lines.append(
                    f"We also have user-uploaded knowledge base related to {descriptions}."
                )
            else:
                lines.append(
                    "We also have user-uploaded documents in the knowledge base."
                )

        if not base_docs and not upload_docs:
            lines.append("Knowledge base contains private documents.")

        return " ".join(lines)