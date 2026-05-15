# build_faiss.py

from loguru import logger
from aira.rag.loader import PDFLoader
from aira.rag.semantic_chunker import SemanticChunker
from aira.rag.embeddings import EmbeddingModel
from aira.rag.vectorstore import FAISSVectorStore
from aira.rag.summarizer import MapReduceSummarizer
from aira.rag.kb_metadata import KBMetadata
from aira.core.config import FAISS_INDEX_PATH, RAG_DOC
from aira.core.dependencies import get_llm
import os


def main():
    logger.info("Starting FAISS index build with semantic chunking + summarization")

    loader = PDFLoader()
    chunker = SemanticChunker()
    embedding_model = EmbeddingModel().get()
    vectorstore = FAISSVectorStore(embedding_model)

    # Load LLM for summarization
    llm = get_llm()
    summarizer = MapReduceSummarizer(llm)
    metadata = KBMetadata()

    all_chunks = []

    # Process each PDF individually so we can summarize per-file
    pdf_files = [
        f for f in os.listdir(RAG_DOC)
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        logger.error(f"No PDFs found in {RAG_DOC}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAG_DOC, pdf_file)
        logger.info(f"Processing: {pdf_file}")

        # Load pages
        pages = loader.load(pdf_path)
        logger.info(f"  Loaded {len(pages)} pages")

        # Semantic chunk
        chunks = chunker.split(pages)
        logger.info(f"  Created {len(chunks)} chunks")

        # Map-reduce summarize
        description = summarizer.summarize(chunks, pdf_file)

        # Save description to metadata
        metadata.add_base_doc(pdf_file, description)

        all_chunks.extend(chunks)

    logger.info(f"Total chunks across all docs: {len(all_chunks)}")

    # Build and save FAISS index
    vectorstore.build(all_chunks)
    vectorstore.save(FAISS_INDEX_PATH)

    logger.info("FAISS index built and saved")
    logger.info("Metadata summary:")
    for fname, desc in metadata.get_base_docs().items():
        logger.info(f"  {fname} → {desc}")


if __name__ == "__main__":
    main()