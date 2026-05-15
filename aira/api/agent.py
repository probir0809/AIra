# api/agent.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from loguru import logger

from aira.agents.tool_agent import ToolAgent
from aira.core.dependencies import get_llm

router = APIRouter(prefix="/v1/agent", tags=["Agent"])

# Cached singleton — agent is expensive to initialize
# (loads retriever, builds tools, wires LLM)
_agent: ToolAgent = None


def get_agent() -> ToolAgent:
    global _agent
    if _agent is None:
        logger.info("Initializing ToolAgent singleton...")
        _agent = ToolAgent(get_llm())
    return _agent


# ── Request / Response models ────────────────────────────────────────────────

class AgentRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the structure of a scientific article?"
            }
        }


class ToolCall(BaseModel):
    tool: str
    input: str
    output: str


class AgentResponse(BaseModel):
    question: str
    answer: str
    tool_calls: List[ToolCall]
    tools_used: List[str]


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=AgentResponse)
def agent_chat(request: AgentRequest):
    """
    Agentic chat endpoint.

    The agent decides which tools to use based on the question:
    - document_search  → private knowledge base (PDFs)
    - web_search       → current events / general knowledge
    - calculator       → math and numeric computation

    Returns the final answer plus a full trace of every tool call made.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Agent chat request: '{request.question}'")

    agent = get_agent()
    result = agent.run(request.question)

    # Build ToolCall models from raw dicts
    tool_calls = [
        ToolCall(
            tool=tc["tool"],
            input=str(tc["input"]),
            output=tc["output"],
        )
        for tc in result.get("tool_calls", [])
    ]

    # Deduplicated list of tool names used — useful for frontend trace panel
    tools_used = list(dict.fromkeys(tc.tool for tc in tool_calls))

    return AgentResponse(
        question=request.question,
        answer=result["answer"],
        tool_calls=tool_calls,
        tools_used=tools_used,
    )