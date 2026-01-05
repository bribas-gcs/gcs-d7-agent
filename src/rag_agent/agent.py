from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_COLLECTION = "d7-visa-knowledge"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Defina a variável de ambiente OPENAI_API_KEY para usar o agente."
        )
    return key


def load_documents(data_dir: Path) -> List[Document]:
    """Load PDF and Word documents from the data directory."""
    ensure_directory(data_dir)
    loaders = [
        DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(data_dir), glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(str(data_dir), glob="**/*.doc", loader_cls=Docx2txtLoader),
    ]

    documents: List[Document] = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def build_vector_store(
    documents: Sequence[Document],
    embedding_model: Optional[object] = None,
    persist_dir: Optional[Path] = None,
) -> Optional[Chroma]:
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    split_docs = splitter.split_documents(documents)

    embedding = embedding_model or OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = str(persist_dir) if persist_dir else None

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        collection_name=DEFAULT_COLLECTION,
        persist_directory=persist_directory,
    )
    if persist_directory:
        vector_store.persist()
    return vector_store


def build_retriever(
    data_dir: Path,
    *,
    embedding_model: Optional[object] = None,
    persist_dir: Optional[Path] = None,
    reindex: bool = False,
) -> Optional[object]:
    """
    Create or load the retriever for local knowledge.
    """
    persist_path = ensure_directory(persist_dir) if persist_dir else None
    has_index = persist_path and (persist_path / "chroma.sqlite3").exists()
    can_reuse = bool(has_index and not reindex)

    if can_reuse:
        return Chroma(
            collection_name=DEFAULT_COLLECTION,
            embedding_function=embedding_model
            or OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=str(persist_path),
        ).as_retriever(search_kwargs={"k": 4})

    documents = load_documents(data_dir)
    store = build_vector_store(documents, embedding_model, persist_path)
    return store.as_retriever(search_kwargs={"k": 4}) if store else None


def build_search_tool() -> Optional[TavilySearchResults]:
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        return TavilySearchResults(max_results=3)
    return None


def _format_context(documents: Iterable[Document]) -> str:
    docs = list(documents)
    if not docs:
        return "Nenhum documento local encontrado no diretório de dados."

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Fonte desconhecida")
        formatted.append(f"Fonte: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)


def _format_news(results: Optional[Sequence[dict]]) -> str:
    if not results:
        return "Pesquisa de notícias não configurada ou sem resultados."
    lines = []
    for item in results:
        title = item.get("title") or "Notícia"
        content = item.get("content") or item.get("snippet") or ""
        url = item.get("url")
        suffix = f" ({url})" if url else ""
        lines.append(f"- {title}: {content}{suffix}")
    return "\n".join(lines)


@dataclass
class VisaAgent:
    data_dir: Path
    persist_dir: Optional[Path] = None
    reindex: bool = False
    llm: Optional[BaseChatModel] = None
    embedding: Optional[object] = None
    search_client: Optional[object] = None

    def __post_init__(self) -> None:
        if self.llm is None:
            ensure_openai_api_key()
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            )

        if self.embedding is None:
            ensure_openai_api_key()
            self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.search_client = self.search_client or build_search_tool()

        self.retriever = build_retriever(
            self.data_dir,
            embedding_model=self.embedding,
            persist_dir=self.persist_dir,
            reindex=self.reindex,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Você é um especialista no Visto D7 de Portugal. "
                        "Use rigorosamente o contexto fornecido e, quando disponível, notícias recentes. "
                        "Forneça respostas em português europeu, cite fontes quando possível "
                        "e destaque passos práticos claros."
                    ),
                ),
                (
                    "user",
                    (
                        "Pergunta do usuário: {question}\n\n"
                        "Contexto local (documentos):\n{context}\n\n"
                        "Pesquisas recentes:\n{news}\n\n"
                        "Resuma com precisão e inclua avisos importantes."
                    ),
                ),
            ]
        )

    def _fetch_news(self, question: str) -> str:
        if not self.search_client:
            return "Pesquisa de notícias não configurada ou indisponível."
        try:
            query = f"{question} visto D7 Portugal notícias recentes requisitos atualizados"
            results = self.search_client.invoke(query)
            return _format_news(results)
        except Exception as exc:  # pragma: no cover - log unexpected issues
            return f"Falha ao pesquisar notícias: {exc}"

    def _retrieve_context(self, question: str) -> List[Document]:
        if not self.retriever:
            return []

        # LangChain retrievers in 0.2+ are Runnables (use invoke). Legacy retrievers expose get_relevant_documents.
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(question)
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(question)
        return []

    def answer_question(self, question: str) -> str:
        context_docs = self._retrieve_context(question)
        context = _format_context(context_docs)
        news = self._fetch_news(question)
        messages = self.prompt.format_messages(
            question=question,
            context=context,
            news=news,
        )
        response = self.llm.invoke(messages)
        if isinstance(response, AIMessage):
            return response.content
        if hasattr(response, "content"):
            return response.content
        return str(response)


def create_agent(
    data_dir: Path,
    persist_dir: Optional[Path] = None,
    reindex: bool = False,
    llm: Optional[BaseChatModel] = None,
    embedding: Optional[object] = None,
    search_client: Optional[object] = None,
) -> VisaAgent:
    """
    Factory helper for easier instantiation and testing.
    """
    return VisaAgent(
        data_dir=data_dir,
        persist_dir=persist_dir,
        reindex=reindex,
        llm=llm,
        embedding=embedding,
        search_client=search_client,
    )
