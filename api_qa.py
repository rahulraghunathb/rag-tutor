"""
Q&A Mode API - Handles question answering with context retrieval
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os
import requests

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Constants
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_TOP_K = 6
LLM_MODEL = None
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# Pydantic Models
class AskRequest(BaseModel):
    question: str = Field(..., description="User question to retrieve context for")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of contexts to retrieve")


class ContextItem(BaseModel):
    rank: int
    source: Optional[str] = None
    page: Optional[int] = None
    text: str


class AskResponse(BaseModel):
    question: str
    top_k: int
    results: List[ContextItem]
    snippet: str
    citations: List[dict]


# Global instances
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model


def _load_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise e
    return _collection


def _get_llm_model() -> str:
    global LLM_MODEL
    if LLM_MODEL is None:
        load_dotenv()
        LLM_MODEL = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
    return LLM_MODEL


def _get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not found in environment")
    return api_key


def _check_llm_health() -> bool:
    """Check if OpenRouter API key is available."""
    load_dotenv()
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _call_openrouter(messages: List[dict], max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """Helper to call OpenRouter API."""
    api_key = _get_api_key()
    model_name = _get_llm_model()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenRouter API call failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service error: {e}")


def _is_qa_related(question: str) -> bool:
    """Use LLM to check if the question is related to networking/security Q&A."""
    if not _check_llm_health():
        return True  # Default to True if LLM config is missing
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a classifier. Determine if the given question is related to networking, security, "
                    "protocols, systems, or technical topics. Respond with only 'YES' or 'NO'."
                ),
            },
            {
                "role": "user",
                "content": f"Is this question related to networking, security, or technical topics? Question: {question}",
            },
        ]
        answer = _call_openrouter(messages, max_tokens=10, temperature=0.1)
        return "YES" in answer.upper()
    except Exception as e:
        print(f"Error checking if question is QA-related: {e}")
        return True  # Default to True on error


def _embed_query(text: str) -> List[float]:
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=False, normalize_embeddings=True)
    return vec[0].tolist()


def _fetch_context(query_embedding: List[float], top_k: int) -> List[ContextItem]:
    collection = _load_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    items: List[ContextItem] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source") if isinstance(meta, dict) else None
        page = meta.get("page") if isinstance(meta, dict) else None
        text = (doc or "").strip()
        items.append(ContextItem(rank=idx, source=source, page=page, text=text))
    return items


def _summarize_with_llm(question: str, contexts: List[ContextItem]) -> str:
    if not _check_llm_health():
        raise HTTPException(status_code=503, detail="LLM service is not configured.")
    
    context_text = "\n\n".join(
        [f"[{c.rank}] Source: {c.source or 'unknown'}{(' · p' + str(c.page)) if c.page else ''}\n{c.text}" for c in contexts]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful networking and security expert. Provide detailed, comprehensive answers using ONLY the provided context. "
                "Include relevant technical details, examples, and explanations. "
                "Cite sources inline like [1], [2]. If unsure, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]
    return _call_openrouter(messages, max_tokens=1000, temperature=0.7)


def ask_question(question: str, top_k: int = DEFAULT_TOP_K) -> AskResponse:
    """Main Q&A function"""
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Check if the question is related to Q&A topics using LLM
    if not _is_qa_related(q):
        # Let the LLM answer without database context
        if not _check_llm_health():
            raise HTTPException(status_code=503, detail="LLM service is not configured.")
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question directly and briefly.",
            },
            {
                "role": "user",
                "content": q,
            },
        ]
        answer = _call_openrouter(messages, max_tokens=1000, temperature=0.7)
        prompt = "\n\n---\n\nFeel free to ask me anything related to networking, security, protocols, or technical topics!"
        return AskResponse(question=q, top_k=top_k, results=[], snippet=answer + prompt, citations=[])

    try:
        q_emb = _embed_query(q)
        contexts = _fetch_context(q_emb, top_k)
        answer = _summarize_with_llm(q, contexts)

        citations = [
            {"index": c.rank, "source": c.source, "page": c.page}
            for c in contexts
        ]
        return AskResponse(question=q, top_k=top_k, results=contexts, snippet=answer, citations=citations)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
