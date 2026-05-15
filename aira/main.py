

from fastapi import FastAPI
from aira.api.chat import router as chat_router
from aira.api.health import router as health_router

app = FastAPI(
    title="AIra",
    description="Modular LLM Serving Framework",
    version="0.2.0"
)



app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router)

