import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def _build_and_persist_embeddings(data_loader, embedding_model, embeddings_dir):
    """
    Loads documents, splits them, and builds/persists the ChromaDB vector store.
    """
    documents = data_loader.load()

    if not documents:
        raise ValueError("[Document Loader] No documents loaded. Please check the data laoder. ❌")

    print("[Document Loader] Splitting documents... ⏳")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    print("[Document Loader] Creating/Loading ChromaDB vector store... ⏳")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=embeddings_dir
    )

    print("[Document Loader] ChromaDB initialized and persisted. ✅")
    
    return vectorstore


def initialize_vector_store(data_loader, embedding_model, embeddings_dir):
    if os.path.exists(embeddings_dir) and os.listdir(embeddings_dir):
        print(f"[INITIALIZATION] Found existing ChromaDB at {embeddings_dir}. Loading embeddings... ⏳")
        try:
            vectorstore = Chroma(
                persist_directory=embeddings_dir,
                embedding_function=embedding_model
            )
            print("[INITIALIZATION] ChromaDB loaded successfully. ✅")
        except Exception as e:
            print(f"[INITIALIZATION] Error loading ChromaDB: {e} ❌. Rebuilding embeddings. ⏳")
            vectorstore = _build_and_persist_embeddings(data_loader, embedding_model, embeddings_dir)
    else:
        print("[INITIALIZATION] No existing ChromaDB. Building and persisting embeddings... ⏳")
        vectorstore = _build_and_persist_embeddings(data_loader, embedding_model, embeddings_dir)
        print("[INITIALIZATION] ChromaDB loaded successfully. ✅")

    return vectorstore