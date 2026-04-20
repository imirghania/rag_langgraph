from typing import List, Protocol

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings  # type: ignore


class Chunker(Protocol):
    def split_documents(self, documents: List[Document]) -> List[Document]: ...


class NoSplitChunker:
    """Treats each document as its own chunk — ideal for self-contained Q&A pairs."""

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return documents


class RecursiveChunker:
    """Splits documents recursively by character boundaries."""

    def __init__(self, 
                chunk_size: int = settings.chunk_size, 
                chunk_overlap: int = settings.chunk_overlap):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)


def get_chunker(strategy: str | None = None) -> Chunker:
    strategy = strategy or settings.chunking_strategy
    if strategy == "no_split":
        return NoSplitChunker()
    if strategy == "recursive":
        return RecursiveChunker()
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose 'no_split' or 'recursive'.")
