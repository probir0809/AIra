from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from loguru import logger
from typing import List
import os


class PDFLoader:
    """
    Responsible for loading PDF files and returning LangChain Documents.
    """

    def load(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF file.
        """
        logger.info(f"Loading PDF: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
        return documents

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all PDFs from a directory.
        """
        logger.info(f"Loading PDFs from directory: {directory_path}")

        all_documents: List[Document] = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                full_path = os.path.join(directory_path, filename)
                docs = self.load(full_path)
                all_documents.extend(docs)

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
