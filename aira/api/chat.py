# from fastapi import APIRouter
# from pydantic import BaseModel

# from aira.core.llm_loader import AIraModel
# from aira.chains.basic_chain import BasicChain

# # Create router
# router = APIRouter(prefix="/chat", tags=["Chat"])

# # Initialize model + chain once (important)
# llm = AIraModel().llm
# chain = BasicChain(llm)

# class ChatRequest(BaseModel):
#     question: str


# class ChatResponse(BaseModel):
#     answer: str

# @router.post("/", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     response = chain.run(request.question)
#     return ChatResponse(answer=response)



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
