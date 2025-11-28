import chainlit as cl
from src.rag import get_rag_chain

@cl.on_chat_start
async def start(): 
    loading_message = cl.Message(content="Initializing RAG chain...")
    await loading_message.send()
    rag_chain = get_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    loading_message.content = "RAG chain is ready. You can now ask your questions."
    await loading_message.update()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("rag_chain")
    if chain is None:
        await cl.Message(content="Session expired. Please refresh.").send()
        return
    try:
        msg = cl.Message(content="")
        await msg.send()
        async for chunk in chain.astream(message.content):
            await msg.stream_token(chunk)
        await msg.update()
    except Exception as e:
        await cl.Message(content="Error processing request.").send()