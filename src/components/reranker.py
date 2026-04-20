from transformers import AutoModel, AutoTokenizer

from config.settings import settings  # type: ignore


def initialize_reranker() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(settings.reranking_model_name)
    model = AutoModel.from_pretrained(settings.reranking_model_name)
    return model, tokenizer
