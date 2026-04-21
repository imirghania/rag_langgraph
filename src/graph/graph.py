import re
from typing import Iterator

import torch
import torch.nn.functional as F
from langgraph.graph import END, START, StateGraph

from graph.state import AgentState  # type: ignore
from utils import trim_think_with_regex  # type: ignore


class RAGGraph:
    def __init__(self, colbert_tokenizer, colbert_model, retriever, llm_chain, top_k_reranked_docs=3):
        self.retriever = retriever
        self.llm_chain = llm_chain
        self.colbert_tokenizer = colbert_tokenizer
        self.colbert_model = colbert_model
        self.top_reranked_docs = top_k_reranked_docs
        self.graph = None


    def setup(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents)
        workflow.add_node("generate", self._generate_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()
        print("LangGraph setup complete with reranking node. ✅")


    def _get_token_embeddings(self, text):
        inputs = self.colbert_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.colbert_model(**inputs)

        input_ids = inputs["input_ids"][0]
        keep_indices = (input_ids != self.colbert_tokenizer.cls_token_id) & (
            input_ids != self.colbert_tokenizer.sep_token_id
        )
        return outputs.last_hidden_state[0][keep_indices]


    def _colbert_score(self, query_emb, doc_emb):
        # query_emb: (m, d), doc_emb: (n, d)
        sim = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=2)
        max_sim, _ = sim.max(dim=1)
        return max_sim.sum().item()


    def _retrieve_documents(self, state: AgentState):
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"[RETRIEVAL] Retrieved {len(documents)} documents for question: '{question}'")
        return {"question": question, "documents": documents}


    def _rerank_documents(self, state: AgentState):
        question = state["question"]
        documents = state["documents"]

        if not documents:
            print("[RERANKING] No documents to rerank.")
            return {"documents": []}

        print(f"[RERANKING] Reranking {len(documents)} documents...")

        query_emb = self._get_token_embeddings(question)
        scores = [
            (self._colbert_score(query_emb, self._get_token_embeddings(doc.page_content)), doc)
            for doc in documents
        ]
        reranked = [doc for _, doc in sorted(scores, key=lambda x: x[0], reverse=True)]
        top_docs = reranked[: self.top_reranked_docs]

        print("[RERANKING] Reranking complete. Top documents:")
        for doc in top_docs:
            print(f"  - {doc.page_content[:100]}...")

        return {"documents": top_docs, "question": question}


    def _generate_answer(self, state: AgentState):
        question = state["question"]
        documents = state["documents"]
        history = state.get("history", [])
        context = "\n\n".join([doc.page_content for doc in documents])

        response = self.llm_chain.invoke(
            {"context": context, "question": question, "history": history}
        )

        if response.startswith("<think>"):
            response = trim_think_with_regex(response)

        print(f"[GENERATION] Answer generated for: '{question}'")
        return {"answer": response}


    def query(self, question: str, history: list | None = None) -> Iterator[str]:
        """Stream answer tokens for the given question."""
        print(f"[QUERY] Streaming answer for: '{question}'")

        state: dict = {"question": question, "documents": [], "answer": ""}
        state.update(self._retrieve_documents(state))
        state.update(self._rerank_documents(state))

        context = "\n\n".join([doc.page_content for doc in state["documents"]])

        buffer = ""
        think_done = False

        for chunk in self.llm_chain.stream(
            {"context": context, "question": question, "history": history or []}
        ):
            if think_done:
                yield chunk
            else:
                buffer += chunk
                if buffer.startswith("<think>"):
                    if "</think>" in buffer:
                        remainder = re.sub(
                            r"<think>.*?</think>", "", buffer, flags=re.DOTALL
                        ).lstrip()
                        think_done = True
                        if remainder:
                            yield remainder
                else:
                    think_done = True
                    yield buffer

        print("[QUERY] Finished streaming. ✅")
