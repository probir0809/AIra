

from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from aira.api.chat import router as chat_router
from aira.api.rag import router as rag_router
from aira.api.health import router as health_router
from aira.api.rerank import router as rerank_router
from aira.api.documents import router as documents_router
from aira.rag.kb_metadata import KBMetadata
from aira.api.agent import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("AIra server starting up...")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("AIra server shutting down — clearing session upload metadata...")
    KBMetadata().clear_upload_docs()
    logger.info("Session upload metadata cleared")


app = FastAPI(
    title="AIra",
    description="Modular LLM Serving Framework",
    version="0.2.0",
    lifespan=lifespan
)



app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router)

