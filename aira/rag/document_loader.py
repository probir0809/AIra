from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader
)
from typing import List
from langchain.schema import Document
import os


class DocumentLoader:
    """
    Unified loader for PDFs, DOCX, HTML files, and Web URLs.
    """

    @staticmethod
    def load(source: str) -> List[Document]:
        """
        Automatically detects file type or URL and loads documents.
        """
        if source.startswith("http"):
            return DocumentLoader._load_web(source)

        ext = os.path.splitext(source)[1].lower()

        if ext == ".pdf":
            return DocumentLoader._load_pdf(source)
        elif ext == ".docx":
            return DocumentLoader._load_docx(source)
        elif ext in [".html", ".htm"]:
            return DocumentLoader._load_html(source)
        else:
            raise ValueError(f"Unsupported document type: {ext}")

    @staticmethod
    def _load_pdf(path: str) -> List[Document]:   # _ infront of method name means private method(encapsulation)
        loader = PyPDFLoader(path)
        return loader.load()

    @staticmethod
    def _load_docx(path: str) -> List[Document]:
        loader = Docx2txtLoader(path)
        return loader.load()

    @staticmethod
    def _load_html(path: str) -> List[Document]:
        loader = UnstructuredHTMLLoader(path)
        return loader.load()

    @staticmethod
    def _load_web(url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()
