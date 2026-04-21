import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    question: str
    documents: Annotated[List[Document], operator.add]
    answer: str
    history: List[BaseMessage]
