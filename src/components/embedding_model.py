import torch
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings  # type: ignore


def get_embedding_model() -> HuggingFaceEmbeddings:
    device = (
        settings.device
        if settings.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
