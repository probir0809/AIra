#rag.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from aira.chains.rag_chain import RAGChain
from aira.core.dependencies import get_rag_chain, get_retriever

router = APIRouter(prefix="/v1/rag", tags=["RAG"])

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=RAGResponse)
def rag_chat(
    request: RAGRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
    retriever = Depends(get_retriever)
):
    documents = retriever.retrieve(request.question)
    answer = rag_chain.run(request.question, documents)

    return {"answer": answer}
