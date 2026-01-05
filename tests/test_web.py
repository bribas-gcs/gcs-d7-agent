from fastapi.testclient import TestClient

from rag_agent.web import create_app


class DummyAgent:
    def __init__(self, answer: str = "ok"):
        self.answer = answer

    def answer_question(self, question: str) -> str:
        return f"{self.answer}:{question}"


def test_chat_endpoint_returns_agent_answer():
    app = create_app(agent=DummyAgent("resposta"))
    client = TestClient(app)

    response = client.post("/api/chat", json={"question": "Olá, agente?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "resposta:Olá, agente?"}


def test_chat_validation_rejects_short_question():
    app = create_app(agent=DummyAgent())
    client = TestClient(app)

    response = client.post("/api/chat", json={"question": "Oi"})

    assert response.status_code == 422


def test_homepage_served_from_static():
    app = create_app(agent=DummyAgent())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Assistente RAG" in response.text
