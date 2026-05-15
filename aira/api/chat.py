# chat.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from aira.chains.basic_chain import BasicChain
from aira.core.dependencies import get_basic_chain

router = APIRouter(prefix="/v1/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    chain: BasicChain = Depends(get_basic_chain)
):
    response = chain.run(request.question)
    return {"answer": response}
