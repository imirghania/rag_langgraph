import asyncio
import threading

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

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
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):
    rag_system = cl.user_session.get("rag_system")

    if rag_system is None:
        await cl.Message(
            content="RAG system not initialized. Please wait or restart the chat."
        ).send()
        return

    history = cl.user_session.get("history", [])

    msg = cl.Message(content="")
    await msg.send()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def run_generator():
        try:
            for token in rag_system.query_langgraph(message.content, history):
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    thread = threading.Thread(target=run_generator, daemon=True)
    thread.start()

    full_answer = ""
    try:
        while True:
            token = await queue.get()
            if token is None:
                break
            full_answer += token
            await msg.stream_token(token)
        await msg.update()
    except asyncio.CancelledError:
        stop_event.set()
        raise
    finally:
        thread.join(timeout=2)

    # Update conversation history, keeping the last max_history_turns pairs
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=full_answer))
    max_messages = settings.max_history_turns * 2
    cl.user_session.set("history", history[-max_messages:])
