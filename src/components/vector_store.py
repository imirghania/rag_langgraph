import os
from typing import Optional

from langchain_chroma import Chroma

from components.chunking import Chunker, get_chunker  # type: ignore


def _build_and_persist_embeddings(data_loader, embedding_model, embeddings_dir, chunker: Chunker):
    documents = data_loader.load()

    if not documents:
        raise ValueError("[VECTOR STORE] No documents loaded. Please check the data loader. ❌")

    print(f"[VECTOR STORE] Chunking {len(documents)} documents with {type(chunker).__name__}... ⏳")
    splits = chunker.split_documents(documents)
    print(f"[VECTOR STORE] {len(splits)} chunks ready. Building ChromaDB... ⏳")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=embeddings_dir,
    )

    print("[VECTOR STORE] ChromaDB initialized and persisted. ✅")
    return vectorstore


def initialize_vector_store(data_loader, embedding_model, embeddings_dir, chunker: Optional[Chunker] = None):
    chunker = chunker or get_chunker()

    if os.path.exists(embeddings_dir) and os.listdir(embeddings_dir):
        print(f"[VECTOR STORE] Found existing ChromaDB at {embeddings_dir}. Loading... ⏳")
        try:
            vectorstore = Chroma(
                persist_directory=embeddings_dir,
                embedding_function=embedding_model,
            )
            print("[VECTOR STORE] ChromaDB loaded successfully. ✅")
        except Exception as e:
            print(f"[VECTOR STORE] Error loading ChromaDB: {e} ❌. Rebuilding... ⏳")
            vectorstore = _build_and_persist_embeddings(data_loader, embedding_model, embeddings_dir, chunker)
    else:
        print("[VECTOR STORE] No existing ChromaDB. Building from scratch... ⏳")
        vectorstore = _build_and_persist_embeddings(data_loader, embedding_model, embeddings_dir, chunker)

    return vectorstore
