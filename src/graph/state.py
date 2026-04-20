import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        documents: List of retrieved documents.
        answer: The final answer from the LLM.
    """
    question: str
    documents: Annotated[List[Document], operator.add]
    answer: str