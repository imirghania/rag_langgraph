from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    model_name: str = "deepseek-r1:1.5b"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    reranking_model_name: str = "colbert-ir/colbertv2.0"
    top_reranked_docs: int = 3
    retriever_k: int = 10
    embeddings_dir: str = "./vector_stores"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    temperature: float = 0.7
    excel_file: str = "data/knowledge_base.xlsx"
    chunking_strategy: str = "no_split"  # "no_split" or "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_history_turns: int = 10  # number of (human, ai) pairs kept per session

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
