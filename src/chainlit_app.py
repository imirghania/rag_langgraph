import chainlit as cl

from components.data_loader import ExcelLoader  # type: ignore
from config.settings import settings  # type: ignore
from rag_system import RAGSystem  # type: ignore

rag_system_instance = None


@cl.on_chat_start
async def start():
    global rag_system_instance
    if rag_system_instance is None:
        await cl.Message(content="Initializing RAG system... This may take a moment.").send()
        rag_system_instance = RAGSystem(
            data_loader=ExcelLoader(file_path=settings.excel_file),
            model_name=settings.model_name,
        )
        await cl.Message(content="RAG system initialized. Ask me a question!").send()
    else:
        await cl.Message(content="Ask me a question!").send()

    cl.user_session.set("rag_system", rag_system_instance)


@cl.on_message
async def main(message: cl.Message):
    rag_system = cl.user_session.get("rag_system")

    if rag_system is None:
        await cl.Message(
            content="RAG system not initialized. Please wait or restart the chat."
        ).send()
        return

    msg = cl.Message(content="")
    await msg.send()

    for token in rag_system.query_langgraph(message.content):
        await msg.stream_token(token)

    await msg.send()
