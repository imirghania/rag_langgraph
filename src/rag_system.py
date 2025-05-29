import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.data_loader import ExcelLoader

from src.prompts.rag import prompt_template
from src.state import AgentState


class RAGSystem:
    def __init__(self, 
                data_loader: BaseLoader, 
                model_name: str = "llama2",
                embedding_model_name: str = "all-MiniLM-L6-v2",
                embeddings_dir: str = "./vector_stores"):
        
        self.data_loader = data_loader
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.embeddings_dir = embeddings_dir
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.llm_chain = None
        self.graph = None

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

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        print(f"Initializing Ollama LLM with model: {self.model_name}...")
        self.llm = ChatOllama(model=self.model_name, temperature=0.7)

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
        Sets up the RAG graph using LangGraph for more complex flows (optional but good for extensibility).
        For a simple RAG, this might be overkill, but demonstrates LangGraph usage.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()
        print("LangGraph setup complete.")


    def _retrieve_documents(self, state: AgentState):
        """
        Retrieves documents based on the question.
        """
        question = state["question"]
        documents = self.retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents for question: '{question}'")
        print(f"\nDocuments\n{documents}")
        return {"documents": documents}


    def _generate_answer(self, state: AgentState):
        """
        Generates an answer using the retrieved documents and the LLM.
        """
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join([doc.page_content for doc in documents])

        response = self.llm_chain.invoke({"context": context, "question": question})
        print(f"Generated answer for question: '{question}'")
        return {"answer": response}


    def query_rag_chain(self, question: str) -> str:
        """
        Queries the RAG system using the LangChain expression language chain.
        """
        print(f"Querying LangChain RAG chain with question: '{question}'")
        return self.llm_chain.invoke(question)


    def query_langgraph(self, question: str) -> str:
        """
        Queries the RAG system using the LangGraph.
        """
        print(f"Querying LangGraph with question: '{question}'")
        result = self.graph.invoke({"question": question})
        return result["answer"]


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

        print("\n--- Testing LangGraph ---")
        answer_lg = rag_system.query_langgraph("Deposit not received?")
        print(f"Answer (LangGraph): {answer_lg}")

        # Clean up ChromaDB if needed for fresh start (uncomment if desired)
        # import shutil
        # if os.path.exists("./chroma_db"):
        #     shutil.rmtree("./chroma_db")
        #     print("ChromaDB directory removed.")