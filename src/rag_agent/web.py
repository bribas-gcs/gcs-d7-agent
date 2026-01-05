from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_agent.agent import VisaAgent, create_agent


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "static"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pergunta para o agente D7")


class ChatResponse(BaseModel):
    answer: str


def _default_agent() -> VisaAgent:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    persist_dir = Path(os.getenv("PERSIST_DIR", ".chroma"))
    reindex = os.getenv("REINDEX_ON_STARTUP", "false").lower() in {"1", "true", "yes"}
    return create_agent(data_dir=data_dir, persist_dir=persist_dir, reindex=reindex)


def create_app(agent: Optional[VisaAgent] = None) -> FastAPI:
    """Create a FastAPI instance wired to the visa agent."""

    load_dotenv()
    app = FastAPI(
        title="D7 Visa Agent API",
        description=(
            "API para enviar perguntas ao agente de RAG sobre o Visto D7 de Portugal. "
            "Configure OPENAI_API_KEY e TAVILY_API_KEY (opcional) no ambiente."
        ),
        version="1.0.0",
    )

    app_agent = agent or _default_agent()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home_page() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="Página inicial não encontrada.")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        try:
            answer = app_agent.answer_question(payload.question)
        except Exception as exc:  # pragma: no cover - proteção de produção
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return ChatResponse(answer=answer)

    return app


app = create_app()
