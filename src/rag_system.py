from typing import Iterator

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from components.data_loader import ExcelLoader  # type: ignore  # noqa: F401 (re-exported for callers)
from components.embedding_model import get_embedding_model  # type: ignore
from components.llm import get_llm  # type: ignore
from components.reranker import initialize_reranker  # type: ignore
from components.vector_store import initialize_vector_store  # type: ignore
from config.settings import settings  # type: ignore
from graph.graph import RAGGraph  # type: ignore
from prompts.rag import prompt_template  # type: ignore


class RAGSystem:
    def __init__(self, data_loader: BaseLoader, model_name: str | None = None):
        self.data_loader = data_loader
        self.model_name = model_name or settings.model_name
        self._initialize()

    def _initialize(self):
        print("[INITIALIZATION] Initializing embedding model... ⏳")
        embeddings = get_embedding_model()

        vectorstore = initialize_vector_store(
            self.data_loader, embeddings, settings.embeddings_dir
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})

        print(f"[INITIALIZATION] Initializing LLM: {self.model_name}... ⏳")
        llm = get_llm(self.model_name)

        print("[INITIALIZATION] Initializing ColBERT reranker... ⏳")
        colbert_model, colbert_tokenizer = initialize_reranker()
        print("[INITIALIZATION] ColBERT initialized. ✅")

        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        self.rag_graph = RAGGraph(
            colbert_tokenizer=colbert_tokenizer,
            colbert_model=colbert_model,
            retriever=retriever,
            llm_chain=llm_chain,
            top_k_reranked_docs=settings.top_reranked_docs,
        )
        self.rag_graph.setup()

    def query_langgraph(self, question: str) -> Iterator[str]:
        return self.rag_graph.query(question)
