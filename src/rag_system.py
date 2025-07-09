import os
from typing import Iterator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.data_loader import ExcelLoader

from src.prompts.rag import prompt_template
from src.state import AgentState
from src.utils import trim_think_with_regex


class RAGSystem:
    def __init__(self,
                data_loader: BaseLoader,
                model_name: str = "llama2",
                embedding_model_name: str = "all-MiniLM-L6-v2",
                reranking_model_name: str = "colbert-ir/colbertv2.0",
                embeddings_dir: str = "./vector_stores"):

        self.data_loader = data_loader
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.reranking_model_name = reranking_model_name
        self.embeddings_dir = embeddings_dir
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.llm_chain = None
        self.graph = None
        self.colbert_tokenizer = None 
        self.colbert_model = None 

        self._initialize_components()


    def _initialize_components(self):
        print("Initializing Embedding Model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cuda'}, # 'cpu' if GPU not available -> config
            encode_kwargs={'normalize_embeddings': True}
        )

        if os.path.exists(self.embeddings_dir) and os.listdir(self.embeddings_dir):
            print(f"Found existing ChromaDB at {self.embeddings_dir}. Loading embeddings...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.embeddings_dir,
                    embedding_function=self.embeddings
                )
                print("ChromaDB loaded successfully.")
            except Exception as e:
                print(f"Error loading ChromaDB: {e}. Rebuilding embeddings.")
                self._build_and_persist_embeddings()
        else:
            print(f"No existing ChromaDB. Building and persisting embeddings...")
            self._build_and_persist_embeddings()

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        print(f"Initializing Ollama LLM with model: {self.model_name}...")
        self.llm = ChatOllama(model=self.model_name, temperature=0.7)

        print("Initializing Colbert model for reranking...")
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(self.reranking_model_name)
        self.colbert_model = AutoModel.from_pretrained(self.reranking_model_name)
        print("Colbert model initialized.")


        self._setup_llm_chain()
        self._setup_langgraph()


    def _build_and_persist_embeddings(self):
        """
        Loads documents, splits them, and builds/persists the ChromaDB vector store.
        """
        print("Loading documents from Excel using ExcelLoader...")
        documents = self.data_loader.load()

        if not documents:
            raise ValueError(f"No documents loaded. Please check the data laoder.")

        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        print("Creating/Loading ChromaDB vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.embeddings_dir
        )

        print("ChromaDB initialized and persisted")


    def _setup_llm_chain(self):
        """
        Sets up the basic RAG chain using LangChain.
        """
        self.prompt = ChatPromptTemplate.from_template(prompt_template)

        self.llm_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("LangChain RAG chain setup complete.")


    def _setup_langgraph(self):
        """
        Sets up the RAG graph using LangGraph for more complex flows.
        Added a rerank node to the workflow.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents) 
        workflow.add_node("generate", self._generate_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank") 
        workflow.add_edge("rerank", "generate") 
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()
        print("LangGraph setup complete with reranking node.")


    def _get_token_embeddings(self, text):
        """
        Get token-level embeddings, ignore [CLS] and [SEP].
        """
        inputs = self.colbert_tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.colbert_model(**inputs)
        
        
        input_ids = inputs['input_ids'][0] 
        keep_indices = (input_ids != self.colbert_tokenizer.cls_token_id) & (input_ids != self.colbert_tokenizer.sep_token_id) 
        
        return outputs.last_hidden_state[0][keep_indices]


    def _colbert_score(self, query_emb, doc_emb): #
        """
        Calculates the Colbert score between query and document embeddings.
        query_emb: (m, d), doc_emb: (n, d)
        """
        # unsqueeze for broadcasting to (m, 1, d) and (1, n, d) respectively
        sim = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=2)  # (m, n)

        max_sim, _ = sim.max(dim=1)  # (m,)

        return max_sim.sum().item()


    def _retrieve_documents(self, state: AgentState):
        """
        Retrieves documents based on the question.
        """
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents for question: '{question}'")
        # print(f"\nDocuments\n{documents}") # Optional: uncomment to see retrieved docs
        return {"question": question, "documents": documents} # Pass question along


    def _rerank_documents(self, state: AgentState):
        """
        Reranks retrieved documents using the Colbert model.
        """
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            print("No documents to rerank.")
            return {"documents": []}
        
        print(f"Reranking {len(documents)} documents for question: '{question}'...")
        
        query_emb = self._get_token_embeddings(question)
        
        scores = []
        for doc in documents:
            doc_emb = self._get_token_embeddings(doc.page_content)
            score = self._colbert_score(query_emb, doc_emb)
            scores.append((score, doc))
        
        # Sort documents by score in descending order
        reranked_documents = [doc for score, doc in sorted(scores, key=lambda x: x[0], reverse=True)] #
        
        # Optionally, select top K documents after reranking
        top_k_reranked_documents = reranked_documents[:3] # Keeping top 3 for generation
        
        print(f"Reranking complete. Top documents (after reranking):")
        for doc in top_k_reranked_documents:
            print(f"- {doc.page_content[:100]}...") # Print first 100 chars of top docs

        return {"documents": top_k_reranked_documents, "question": question}


    def _generate_answer(self, state: AgentState):
        """
        Generates an answer using the retrieved documents and the LLM.
        """
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join([doc.page_content for doc in documents])

        response = self.llm_chain.invoke({"context": context, "question": question})
        
        if response.startswith("<think>"):
            response = trim_think_with_regex(response)
        
        print(f"Generated answer for question: '{question}'")
        return {"answer": response}


    def query_langgraph(self, question: str) -> Iterator[str]:
        """
        Queries the RAG system using the LangChain expression language chain.
        """
        print(f"Querying LangGraph with question: '{question}' for streaming...")

        full_answer = ""
        for state_update in self.graph.stream({"question": question}):
            if "generate" in state_update:
                chunk = state_update["generate"]["answer"]
                # print("=*"*50)
                # print(chunk)
                # print("=*"*50)
                full_answer += chunk
                yield chunk

        print("Finished streaming from LangGraph.")


if __name__ == "__main__":
    # Ensure data/multi_sheet_data.xlsx exists from the data_loader.py example
    # You might want to run data_loader.py directly once to create it.
    # dummy_excel_path = "data/multi_sheet_data.xlsx"
    dummy_excel_path = "data/valor_knowledge_base.xlsx"
    if not os.path.exists(dummy_excel_path):
        print(f"Please run 'python data_loader.py' first to create {dummy_excel_path}")
    else:
        data_loader = ExcelLoader(file_path=dummy_excel_path)
        rag_system = RAGSystem(
            data_loader=data_loader,
            model_name="deepseek-r1:1.5b"
        )

        print("\n--- Testing LangGraph with Reranking ---")
        # To observe reranking, you might need a query that returns
        # more documents than are eventually used for generation.
        # Ensure your retriever's k is set appropriately (e.g., k=10)
        # in _initialize_components.
        for chunk in rag_system.query_langgraph("What are the health benefits of meditation?"):
            print(chunk, end="", flush=True)
        print("\n")

        print("\n--- Testing another query ---")
        for chunk in rag_system.query_langgraph("How to deposit cash?"):
            print(chunk, end="", flush=True)
        print("\n")

        # Clean up ChromaDB if needed for fresh start (uncomment if desired)
        # import shutil
        # if os.path.exists("./vector_stores"): # Changed from chroma_db to vector_stores
        #     shutil.rmtree("./vector_stores")
        #     print("Vector store directory removed.")