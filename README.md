# Agente RAG para o Visto D7 (Portugal)

Este projeto cria um agente RAG em Python usando LangChain + OpenAI para responder dúvidas sobre o Visto D7 de Portugal. Ele lê arquivos PDF ou Word presentes no diretório `data/` e, opcionalmente, consulta notícias recentes para complementar as respostas.

## Pré-requisitos

- Python 3.11+
- Variáveis de ambiente:
  - `OPENAI_API_KEY` – chave da API OpenAI.
  - `TAVILY_API_KEY` – **opcional**, habilita buscas de notícias recentes via Tavily.
- Exemplo: copie `.env.example` para `.env` e preencha as chaves antes de rodar o agente.
- O arquivo `.env` é carregado automaticamente (via `python-dotenv`) ao executar `main.py`.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Preparando os dados

1. Coloque arquivos `.pdf`, `.doc` ou `.docx` com conhecimento sobre o Visto D7 dentro do diretório `data/`.
2. Para forçar a reindexação (ex.: após adicionar novos arquivos), use a flag `--reindex`.

## Como executar

Pergunta direta:

```bash
export OPENAI_API_KEY=...  # e opcionalmente TAVILY_API_KEY=...
python main.py --question "Quais os requisitos do Visto D7?"
```

Modo interativo:

```bash
python main.py
```

Parâmetros úteis:

- `--data-dir`: caminho alternativo para os documentos (padrão: `data/`).
- `--persist-dir`: pasta para persistir o índice vetorial (padrão: `.chroma`).
- `--reindex`: força a reindexação dos documentos.

## Testes

Execute a suíte de testes com:

```bash
pytest
```

Os testes usam modelos e embeddings falsos, portanto não precisam de chaves de API nem de internet.

## GitHub Actions

O fluxo de CI (`.github/workflows/ci.yml`) instala as dependências e executa `pytest` em cada push ou pull request, garantindo que o agente permaneça saudável.
