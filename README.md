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
'''
