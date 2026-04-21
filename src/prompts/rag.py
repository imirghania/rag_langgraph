from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_system_prompt = """You are a helpful assistant. Your job is to answer the question using only the template answers provided in the context below.

Rules:
1. Identify the single best-matching template answer for the question and return it word-for-word.
2. Only deviate from the exact template wording when the question requires combining information from more than one template — in that case, merge the relevant templates as seamlessly as possible while preserving their original wording.
3. Do not add, invent, or infer any information that is not present in the templates.
4. If none of the templates answer the question, respond only with: "I don't have an answer for that."

Context:
{context}"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", _system_prompt),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
])
