# Local RAG System with LangChain, LangGraph, Ollama, and Chainlit

A fully local Retrieval-Augmented Generation (RAG) system. All inference runs on your machine via Ollama — no external APIs.

---

## Features

- **Excel knowledge base** — loads `.xlsx` files (multi-sheet, `question`/`answer` columns)
- **ColBERT reranking** — retrieves 10 candidates, reranks and keeps the top 3
- **Token-level streaming** — responses stream word by word in the chat UI
- **Configurable chunking** — `no_split` (default, best for Q&A pairs) or `recursive` character splitting
- **Settings-driven** — all configuration in one place via `.env` or environment variables

---

## Prerequisites

- **Python 3.12+**
- **uv** — `pip install pipx && pipx install uv`
- **Ollama** — install from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull deepseek-r1:1.5b
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd <repo-directory>

# 2. Create virtualenv and install dependencies
uv venv && source .venv/bin/activate
uv sync

# 3. Place your .xlsx knowledge base in data/
#    Each sheet must have 'question' and 'answer' columns.
#    To generate a sample file for testing:
python src/components/data_loader.py
```

---

## Configuration

All settings are in `src/config/settings.py` and can be overridden via a `.env` file at the project root.

| Variable            | Default                    | Description                            |
| ------------------- | -------------------------- | -------------------------------------- |
| `MODEL_NAME`        | `deepseek-r1:1.5b`         | Ollama model to use                    |
| `EXCEL_FILE`        | `data/knowledge_base.xlsx` | Knowledge base path                    |
| `CHUNKING_STRATEGY` | `no_split`                 | `no_split` or `recursive`              |
| `DEVICE`            | `auto`                     | `auto`, `cuda`, or `cpu`               |
| `TOP_RERANKED_DOCS` | `3`                        | Docs passed to the LLM after reranking |
| `EMBEDDINGS_DIR`    | `./vector_stores`          | ChromaDB persistence directory         |

---

## Run

```bash
chainlit run src/chainlit_app.py -w
```

Open [http://localhost:8000](http://localhost:8000). The first run builds and persists ChromaDB embeddings (slow). Subsequent runs load from `./vector_stores/`.
