# 🚀 AIra — Modular LLM Serving & Agentic AI Framework

AIra is a production-oriented framework for building, serving, and extending LLM-based systems — from simple chat APIs to advanced retrieval-augmented and agentic workflows.

The project focuses on clean architecture, extensibility, and real-world engineering practices, rather than notebook-style experimentation.

# 🎯 Project Objectives 
Build a scalable and modular LLM framework

Support multiple stages of LLM systems:

Basic chat inference

Retrieval-Augmented Generation (RAG)

Agentic workflows

Fine-tuning (LoRA / QLoRA)

High-performance serving

Follow industry-grade backend & MLOps patterns:

Dependency Injection

Configuration management

Clear separation of concerns

API-first design

## 🧱 High-Level Architecture
```text
Client
  |
  | HTTP (JSON)
  v
FastAPI
  |
  |-- /v1/chat
  v
Chain Layer
  |
  |-- Prompt Manager
  |
  v
LLM (HuggingFace / Transformers)
```

## 📁 Project Structure

```text
aira/
├── main.py                  # FastAPI entrypoint
├── api/
│   └── chat.py              # Chat API endpoint
├── chains/
│   └── basic_chain.py       # Core LLM inference chain
├── core/
│   ├── llm_loader.py        # Model loading & wrapping
│   ├── prompt_manager.py   # Prompt abstraction
│   ├── config.py            # Centralized configuration
│   └── dependencies.py     # Dependency injection
```


## ✅ Implemented Features
```mermaid
--LLM Inference Core
- Integrated a pretrained LLM (Qwen/Qwen3-0.6B)
Clean abstraction for:
Model loading
Prompt creation
Generation configuration
Output post-processing to remove:
Special tokens
Internal reasoning traces (<think>...</think>)
Deterministic and clean responses
🔹 FastAPI Serving Layer
REST API built using FastAPI
Versioned endpoint:
POST /v1/chat
Request and response validation using Pydantic
Swagger / OpenAPI documentation enabled
Dependency injection for:
Chains
Model lifecycle
Centralized configuration management
```
## 🔌 API Reference
Endpoint
POST /v1/chat

Request Body
{
  "question": "Explain attention mechanism in simple terms."
}

Response
{
  "answer": "The attention mechanism allows the model to focus on the most relevant parts of the input..."
}

🧪 Running the Project Locally
1️⃣ Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Start the API server
uvicorn aira.main:app --reload


Server will be available at:

http://127.0.0.1:8000

4️⃣ API Documentation

Open Swagger UI:

http://127.0.0.1:8000/docs

🧠 Design Philosophy

APIs call chains, chains call models

No global model state

Loose coupling, high cohesion

Easy to extend without refactoring

Production-first mindset

🔮 Planned Enhancements

Retrieval-Augmented Generation (RAG)

Vector databases & document ingestion

Hybrid search (BM25 + vector)

Re-ranking & advanced retrieval techniques

Dataset versioning with DVC

Fine-tuning with LoRA / QLoRA

Optimized serving (vLLM)

Containerization & CI/CD
