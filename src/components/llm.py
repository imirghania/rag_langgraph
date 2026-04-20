from langchain_ollama import ChatOllama

from config.settings import settings  # type: ignore


def get_llm(model_name: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model_name or settings.model_name,
        temperature=settings.temperature,
    )
