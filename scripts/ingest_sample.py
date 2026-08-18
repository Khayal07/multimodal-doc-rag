"""CLI helper: ingest a local PDF and optionally ask a question.

Usage:
    python scripts/ingest_sample.py ingest path/to/file.pdf
    python scripts/ingest_sample.py query "your question" [--top-k 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.services import build_services  # noqa: E402


def _services():
    return build_services(get_settings())


def _cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    report = _services().ingestion.ingest(path)
    print(
        f"Indexed '{report.document}': {report.pages} pages, "
        f"{report.chunks_indexed} chunks, elements={report.element_counts}"
    )
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    services = _services()
    result = services.pipeline.answer(args.question, top_k=args.top_k)
    print(result.answer)
    for citation in result.citations:
        print(f"  {citation}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal RAG CLI helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse and index a local PDF.")
    ingest.add_argument("file", type=str, help="Path to the PDF file.")
    ingest.set_defaults(func=_cmd_ingest)

    query = sub.add_parser("query", help="Ask a question against the index.")
    query.add_argument("question", type=str, help="The question to answer.")
    query.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve.")
    query.set_defaults(func=_cmd_query)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
