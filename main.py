"""
Main FastAPI Application - Networking RAG System
Combines Q&A and Quiz modes with separate module imports
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
# Import Q&A module
from api_qa import (
    AskRequest,
    AskResponse,
    ask_question,
    _load_model,
    _load_collection,
    _check_ollama_health
)

# Import Quiz module
from api_quiz import (
    QuizRequest,
    QuizResponse,
    QuizCheckRequest,
    QuizCheckResponse,
    generate_quiz,
    check_quiz_answer,
    get_hardcoded_topics
)

# Check Ollama status at startup
ollama_status = _check_ollama_health()
print(f"Starting Networking RAG System...")
print(f"Loading Ollama model === {os.getenv('OLLAMA_MODEL')}")
print(f"Ollama Model Status: {'RUNNING' if ollama_status else 'NOT AVAILABLE'}")
print(f"Vector Database: LOADED")
print(f"Server will start on http://127.0.0.1:8000")
print("=" * 50)


# Initialize FastAPI app
app = FastAPI(title="Networking RAG")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        _load_model()
        _load_collection()
        ollama_ok = _check_ollama_health()
        return {
            "status": "ok" if ollama_ok else "warning", 
            "ollama": "ok" if ollama_ok else "not available",
            "embedding_model": "ok",
            "vector_db": "ok"
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/")
def root_ui():
    """Serve the main UI."""
    index_path = Path("templates/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


@app.get("/quiz/topics")
def get_topics():
    """Get list of hardcoded topics for quiz generation."""
    return {"topics": get_hardcoded_topics()}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    """Q&A endpoint - Answer questions with context from the database."""
    return ask_question(payload.question, payload.top_k)


@app.post("/quiz/generate", response_model=QuizResponse)
def generate_quiz_endpoint(payload: QuizRequest):
    """Quiz generation endpoint - Generate quiz questions."""
    return generate_quiz(payload.topic, payload.question_type, payload.count)


@app.post("/quiz/check", response_model=QuizCheckResponse)
async def check_quiz_endpoint(payload: QuizCheckRequest):
    """Quiz checking endpoint - Check user answers with web citations."""
    return await check_quiz_answer(payload.question_id, payload.user_answer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
