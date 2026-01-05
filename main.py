from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_agent.agent import create_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agente RAG especializado no Visto D7 de Portugal."
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Pergunta para o agente. Se omitida, será iniciado um modo interativo.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Diretório onde estão os arquivos PDF/DOC/DOCX com conhecimento local.",
    )
    parser.add_argument(
        "--persist-dir",
        default=".chroma",
        type=Path,
        help="Diretório para persistir o índice vetorial.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Força a reindexação dos documentos.",
    )
    return parser.parse_args()


def interactive_loop(agent) -> None:
    print("Digite sua pergunta sobre o Visto D7 (ou 'sair' para encerrar):")
    while True:
        question = input("> ").strip()
        if not question or question.lower() in {"sair", "exit", "quit"}:
            break
        answer = agent.answer_question(question)
        print("\nResposta:\n")
        print(answer)
        print("\n" + "-" * 80 + "\n")


def main() -> None:
    load_dotenv()
    args = parse_args()
    agent = create_agent(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        reindex=args.reindex,
    )

    if args.question:
        print(agent.answer_question(args.question))
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()
