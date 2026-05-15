# dependencies.py
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from aira.core.llm_loader import AIraModel
from aira.chains.basic_chain import BasicChain
from aira.chains.rag_chain import RAGChain
from aira.rag.retriever import VectorRetriever
from aira.rag.vectorstore import FAISSVectorStore
from aira.rag.embeddings import EmbeddingModel
from aira.rag.reranker import CrossEncoderReranker
from aira.rag.bm25_retriever import BM25Retriever
from aira.rag.hybrid_retriever import HybridRetriever
from aira.core.config import FAISS_INDEX_PATH, RETRIEVER_TOP_K
from loguru import logger

# ── Permanent singletons (survive the entire process lifetime) ──────────────
_llm = None
_basic_chain = None
_rag_chain = None
_retriever = None           # raw FAISS retriever over permanent index
_bm25_retriever = None      # BM25 over permanent index
_hybrid_retriever = None    # full pipeline: hybrid + reranker
_reranker = None

# ── Session store (in-memory only, never written to disk) ───────────────────
# Holds chunks from PDFs uploaded during the current server process.
# Wiped automatically when uvicorn stops — no manual cleanup needed.
_session_vectorstore: Optional[FAISS] = None   # raw LangChain FAISS object
_session_bm25_docs: List[Document] = []        # parallel list for BM25


# ────────────────────────────────────────────────────────────────────────────
# LLM / chains
# ────────────────────────────────────────────────────────────────────────────

def get_llm():
    global _llm
    if _llm is None:
        _llm = AIraModel().llm
    return _llm


def get_basic_chain():
    global _basic_chain
    if _basic_chain is None:
        _basic_chain = BasicChain(get_llm())
    return _basic_chain


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain(get_llm())
    return _rag_chain


# ────────────────────────────────────────────────────────────────────────────
# Permanent retrieval pipeline
# ────────────────────────────────────────────────────────────────────────────

def _get_vectorstore() -> FAISSVectorStore:
    """Loads the permanent FAISS vectorstore from disk."""
    embedding = EmbeddingModel().get()
    vectorstore = FAISSVectorStore(embedding)
    vectorstore.load(FAISS_INDEX_PATH)
    return vectorstore


def get_faiss_retriever() -> VectorRetriever:
    """Raw FAISS retriever over the permanent index."""
    global _retriever
    if _retriever is None:
        vectorstore = _get_vectorstore()
        _retriever = VectorRetriever(vectorstore, k=RETRIEVER_TOP_K)
    return _retriever


def get_bm25_retriever() -> BM25Retriever:
    """BM25 retriever built from the permanent FAISS docstore."""
    global _bm25_retriever
    if _bm25_retriever is None:
        vectorstore = _get_vectorstore()
        docstore = vectorstore.vectorstore.docstore
        all_docs = list(docstore._dict.values())
        _bm25_retriever = BM25Retriever(documents=all_docs, k=RETRIEVER_TOP_K)
    return _bm25_retriever


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def get_retriever() -> "_RerankedHybrid":
    """
    Full permanent pipeline:
    FAISS top-20 + BM25 top-20 → RRF → ~30 candidates → CrossEncoder top-5

    Session chunks are NOT included here — they are merged inside
    _RerankedHybrid.retrieve() at query time so the permanent cache
    is never invalidated by uploads.
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        hybrid = HybridRetriever(
            faiss_retriever=get_faiss_retriever(),
            bm25_retriever=get_bm25_retriever(),
        )
        _hybrid_retriever = _RerankedHybrid(hybrid, get_reranker())
    return _hybrid_retriever


# ────────────────────────────────────────────────────────────────────────────
# Session vectorstore — uploaded PDFs (in-memory, process lifetime only)
# ────────────────────────────────────────────────────────────────────────────

def add_to_session_vectorstore(chunks: List[Document], embedding_model) -> None:
    """
    Adds new chunks to the in-memory session vectorstore.
    Called by api/documents.py after chunking an uploaded PDF.
    Never touches disk — data lives only until uvicorn stops.
    """
    global _session_vectorstore, _session_bm25_docs

    new_vs = FAISS.from_documents(chunks, embedding_model)

    if _session_vectorstore is None:
        _session_vectorstore = new_vs
        logger.info("Session vectorstore created")
    else:
        _session_vectorstore.merge_from(new_vs)
        logger.info("Session vectorstore updated — merged new chunks")

    # Keep a flat list for BM25 coverage of session docs
    _session_bm25_docs.extend(chunks)
    logger.info(
        f"Session store now has {len(_session_bm25_docs)} chunks "
        f"across all uploaded documents"
    )


def get_session_vectorstore() -> Optional[FAISS]:
    """Returns the in-memory session FAISS store, or None if nothing uploaded yet."""
    return _session_vectorstore


def get_session_bm25_docs() -> List[Document]:
    """Returns all chunks uploaded this session (for BM25 scoring)."""
    return _session_bm25_docs


def clear_session_vectorstore() -> None:
    """
    Wipes the session store manually.
    Useful for a 'clear uploads' API endpoint or testing.
    Under normal use this happens automatically when uvicorn stops.
    """
    global _session_vectorstore, _session_bm25_docs
    _session_vectorstore = None
    _session_bm25_docs = []
    logger.info("Session vectorstore cleared")


# ────────────────────────────────────────────────────────────────────────────
# _RerankedHybrid — chains permanent + session retrieval then reranks
# ────────────────────────────────────────────────────────────────────────────

class _RerankedHybrid:
    """
    Retrieval pipeline that merges:
      - Permanent index  : HybridRetriever (FAISS + BM25) → RRF
      - Session index    : in-memory FAISS + BM25 over session docs
    Then reranks the combined candidates with CrossEncoder.

    The permanent hybrid pipeline is cached as a singleton.
    Session results are fetched fresh on every query so uploads
    are reflected immediately without invalidating the permanent cache.
    """

    def __init__(self, hybrid: HybridRetriever, reranker: CrossEncoderReranker):
        self.hybrid = hybrid
        self.reranker = reranker

    def retrieve(self, query: str) -> List[Document]:
        # 1. Permanent pipeline — FAISS + BM25 + RRF
        candidates = self.hybrid.retrieve(query)
        logger.info(f"Permanent pipeline returned {len(candidates)} candidates")

        # 2. Session pipeline — only runs if something was uploaded
        session_vs = get_session_vectorstore()
        session_docs = get_session_bm25_docs()

        if session_vs is not None:
            # Dense search over session chunks
            session_retriever = session_vs.as_retriever(
                search_kwargs={"k": RETRIEVER_TOP_K}
            )
            session_faiss_docs = session_retriever.get_relevant_documents(query)
            logger.info(f"Session FAISS returned {len(session_faiss_docs)} chunks")

            # Sparse BM25 search over session chunks
            if session_docs:
                session_bm25 = BM25Retriever(documents=session_docs, k=RETRIEVER_TOP_K)
                session_bm25_docs = session_bm25.retrieve(query)
                logger.info(f"Session BM25 returned {len(session_bm25_docs)} chunks")
            else:
                session_bm25_docs = []

            # Combine session results with permanent candidates
            # Dedup by content prefix to avoid double-counting
            seen = {doc.page_content[:80] for doc in candidates}
            for doc in session_faiss_docs + session_bm25_docs:
                if doc.page_content[:80] not in seen:
                    candidates.append(doc)
                    seen.add(doc.page_content[:80])

            logger.info(f"Combined candidate pool: {len(candidates)} chunks")

        # 3. Rerank the full combined pool
        final = self.reranker.rerank(query, candidates)
        logger.info(f"Final pipeline output: {len(final)} documents")
        return final