from fastapi import APIRouter
from pydantic import BaseModel

from aira.core.llm_loader import AIraModel
from aira.chains.basic_chain import BasicChain

# Create router
router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize model + chain once (important)
llm = AIraModel().llm
chain = BasicChain(llm)

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = chain.run(request.question)
    return ChatResponse(answer=response)

