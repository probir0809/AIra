# api/rerank.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from aira.core.dependencies import get_retriever, get_faiss_retriever, get_bm25_retriever  # ← all three

router = APIRouter(prefix="/v1/rerank", tags=["Rerank"])


class RerankRequest(BaseModel):
    question: str


class DocumentResult(BaseModel):
    content: str
    source: str
    score_position: int


class RerankResponse(BaseModel):
    question: str
    total_docs: int
    documents: List[DocumentResult]


@router.post("/test", response_model=RerankResponse)
def test_rerank(
    request: RerankRequest,
    retriever=Depends(get_retriever)
):
    """
    Full pipeline:
    FAISS + BM25 → RRF → CrossEncoder → top-5
    """
    documents = retriever.retrieve(request.question)

    return RerankResponse(
        question=request.question,
        total_docs=len(documents),
        documents=[
            DocumentResult(
                content=doc.page_content[:500],
                source=doc.metadata.get("source", "unknown"),
                score_position=idx + 1
            )
            for idx, doc in enumerate(documents)
        ]
    )


@router.post("/compare", response_model=dict)
def compare_retrievers(
    request: RerankRequest,
    faiss_retriever=Depends(get_faiss_retriever),    # ← raw FAISS only
    bm25_retriever=Depends(get_bm25_retriever),      # ← raw BM25 only
    full_retriever=Depends(get_retriever)            # ← full pipeline
):
    """
    Side-by-side comparison of all three stages.
    """
    faiss_docs = faiss_retriever.retrieve(request.question)
    bm25_docs = bm25_retriever.retrieve(request.question)
    final_docs = full_retriever.retrieve(request.question)

    return {
        "question": request.question,
        "faiss_top5": [
            {"position": i + 1, "source": d.metadata.get("source", ""), "preview": d.page_content[:200]}
            for i, d in enumerate(faiss_docs[:5])
        ],
        "bm25_top5": [
            {"position": i + 1, "source": d.metadata.get("source", ""), "preview": d.page_content[:200]}
            for i, d in enumerate(bm25_docs[:5])
        ],
        "final_top5": [
            {"position": i + 1, "source": d.metadata.get("source", ""), "preview": d.page_content[:200]}
            for i, d in enumerate(final_docs[:5])
        ],
    }