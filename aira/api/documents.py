# api/documents.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger
import shutil
import os
from aira.rag.loader import PDFLoader
from aira.rag.semantic_chunker import SemanticChunker
from aira.rag.embeddings import EmbeddingModel
from aira.rag.summarizer import MapReduceSummarizer
from aira.rag.kb_metadata import KBMetadata
from aira.core.config import RAG_DOC
from aira.core.dependencies import get_llm
import aira.core.dependencies as deps

router = APIRouter(prefix="/v1/documents", tags=["Documents"])


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    description: str
    message: str


@router.get("/list")
def list_documents():
    """Lists all documents in both knowledge bases with their descriptions."""
    metadata = KBMetadata()
    base_docs = metadata.get_base_docs()
    upload_docs = metadata.get_upload_docs()

    return {
        "base_knowledge_base": [
            {"filename": k, "description": v}
            for k, v in base_docs.items()
        ],
        "user_uploads": [
            {"filename": k, "description": v}
            for k, v in upload_docs.items()
        ],
        "rag_tool_description": metadata.build_rag_tool_description()
    }


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, summarize it, add to session knowledge base.

    Pipeline:
    1. Save PDF to disk (so it can be loaded and chunked)
    2. Semantic chunk
    3. Map-reduce summarize → uploads_metadata.json
    4. Add to in-memory session vectorstore (not saved to disk)
       — chunks are available for retrieval immediately
       — wiped automatically when the server stops
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = os.path.join(RAG_DOC, file.filename)

    try:
        # Step 1 — save file to disk so PDFLoader can read it
        logger.info(f"Saving uploaded file: {file.filename}")
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2 — load pages and semantic chunk
        loader = PDFLoader()
        pages = loader.load(save_path)
        chunker = SemanticChunker()
        new_chunks = chunker.split(pages)

        if not new_chunks:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")

        # Step 3 — map-reduce summarize and save description to metadata
        logger.info(f"Summarizing {file.filename}...")
        llm = get_llm()
        summarizer = MapReduceSummarizer(llm)
        description = summarizer.summarize(new_chunks, file.filename)

        metadata = KBMetadata()
        metadata.add_upload_doc(file.filename, description)
        logger.info(f"Description: {description}")

        # Step 4 — add chunks to in-memory session vectorstore
        # Not merged into the permanent FAISS index on disk.
        # Chunks are searchable immediately but lost when uvicorn stops.
        embedding_model = EmbeddingModel().get()
        deps.add_to_session_vectorstore(new_chunks, embedding_model)
        logger.info(f"Added {len(new_chunks)} chunks to session vectorstore")

        return UploadResponse(
            filename=file.filename,
            chunks_added=len(new_chunks),
            description=description,
            message=f"Successfully added '{file.filename}' to the session knowledge base."
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))