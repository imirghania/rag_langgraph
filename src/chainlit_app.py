import chainlit as cl
import os
from data_loader import ExcelLoader
from rag_system import RAGSystem

# Initialize your RAG system globally (or handle lazy loading)
# Make sure data/dummy_data.xlsx exists from previous steps
# EXCEL_FILE = "data/dummy_data.xlsx"
EXCEL_FILE = "data/valor_knowledge_base.xlsx"
OLLAMA_MODEL = "deepseek-r1:1.5b" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# Global RAG system instance
rag_system_instance = None

@cl.on_chat_start
async def start():
    global rag_system_instance
    if rag_system_instance is None:
        await cl.Message(content="Initializing RAG system... This may take a moment.").send()
        # Initialize RAGSystem when chat starts
        rag_system_instance = RAGSystem(
            data_loader=ExcelLoader(file_path=EXCEL_FILE),
            model_name=OLLAMA_MODEL
        )
        await cl.Message(content="RAG system initialized. Ask me a question!").send()
    else:
        await cl.Message(content="Ask me a question!").send()

    cl.user_session.set("rag_system", rag_system_instance)


@cl.on_message
async def main(message: cl.Message):
    rag_system = cl.user_session.get("rag_system") # type: RAGSystem
    
    if rag_system is None:
        await cl.Message(content="RAG system not initialized. Please wait or restart the chat.").send()
        return
    
    msg = cl.Message(content="")
    await msg.send()
    
    stream_generator = rag_system.query_langgraph(message.content)
    
    for token in stream_generator:
        await msg.stream_token(token)
    
    await msg.send()