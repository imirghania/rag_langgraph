# Local RAG System with LangChain, LangGraph, Ollama, and Chainlit

This project demonstrates how to build a fully local Retrieval-Augmented Generation (RAG) system using powerful open-source tools. It leverages LangChain for orchestration, LangGraph for defining complex LLM workflows, HuggingFace Embeddings for document vectorization, Ollama for running local Large Language Models (LLMs), and ChromaDB as a local vector store. The system is served via a user-friendly chat interface built with Chainlit.

---

## Project Overview

This RAG system is designed to answer questions based on knowledge extracted from your own local data. The system uses dataloaders that follow the LangChain BaseLoader Interface. This project implemented a data loader that consumes Excel files. Each row in Excel data (with 'question' and 'answer' columns) is treated as a distinct document, and the system intelligently retrieves relevant information to formulate precise answers.

**Key Features:**

- **Fully Local:** All components, from the LLM to the vector database and embeddings, run entirely on your machine, ensuring data privacy and offline capability.
  Excel Data Ingestion: Easily load your proprietary knowledge from Excel files, with support for multiple sheets.
- **Efficient Document Processing:** Documents are chunked and embedded into a local vector store (ChromaDB). The system intelligently persists these embeddings, so they're only generated once, saving time on subsequent runs.
- **LangChain & LangGraph Orchestration:**
  -- **LangChain** handles the LLM pipeline (prompting, generation).
  -- **LangGraph** defines a stateful, multi-step workflow, perfect for extending to more complex conversational agents or tool usage in the future.
- **Local LLM with Ollama:** Integrate with any LLM you've downloaded via Ollama, providing flexibility and control over the model used.
- **Interactive Chat UI with Chainlit:** A sleek web-based chat interface allows for easy interaction with your RAG system, complete with streaming responses for a smoother user experience.

---

### Local Setup and Installation

Follow these steps to get your local RAG system up and running.

#### **Prerequisites**

1. **Python 3.9+:** Ensure you have a compatible Python version installed (This project uses Python 3.12.4).
2. **uv:** Install uv by following the instructions on its GitHub page or using pipx:

```bash
pip install pipx
pipx install uv
```

3. **Ollama:** Download and install Ollama from ollama.com.
   After installation, pull a local LLM model. For example, to pull Llama 2 (7B parameter version):

```bash
ollama pull <model-name>
```

(Note: The project defaults to gemma:2b in rag_system.py and chainlit_app.py for faster local inference, but you can change it.)(Note: The project defaults to gemma:2b in rag_system.py and chainlit_app.py for faster local inference, but you can change it.)

#### 1. Clone the Repository

```bash
git clone <repo-url>
cd <repo-directory>
```

#### 2. Set up Virtual Environment and Install Dependencies

a. Create and Activate Virtual Environment

```bash
uv venv
```

Activate the virtual environment:

- macOS / Linux:

```bash
source .venv/bin/activate
```

- Windows (Command Prompt):

```bash
.venv\Scripts\activate.bat
```

b. install the dependencies

```bash
uv sync
```

#### 3. Prepare Your Data

The system expects your knowledge base to be in `/data` directory. You can store your data in any file format as long as you create a suitable data loader to consume it. The data I used is in Excel files (.xlsx) with specific column names.

Create a directory named data in your project's root.
Inside the data directory, place your Excel files (e.g., my_knowledge.xlsx).
Each sheet in your Excel file must have columns named question and answer.
(Optional: You can have other columns like tag, but the loader specifically uses question and answer for content.)

You may generate a dummy Excel file for testing purposes:

for that run the data_loader.py script once to create a sample data/multi_sheet_data.xlsx file:

```bash
python src/data_loader.py
```

#### 4. Run the Chainlit Application

Make sure your virtual environment is active (see step 2), then navigate to your project's root directory in your terminal and start the Chainlit app:

```bash
chainlit run src/chainlit_app.py -w
# or
PYTHONPATH=$PYTHONPATH:. chainlit run src/chainlit_app.py -w
```

#### 5. Access the UI

Open your web browser and go to http://localhost:8000 (or the address shown in your terminal if a different port is used).

You'll see a chat interface. The first time you load the application, it will take some time to initialize the RAG system, load your Excel data, and generate/persist embeddings. Subsequent runs will be much faster as they'll load the existing embeddings.
