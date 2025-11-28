import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR

def get_rag_chain() -> RunnablePassthrough:
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(VECTOR_STORE_DIR):
        raise FileNotFoundError(f"Vector store directory '{VECTOR_STORE_DIR}' does not exist. Please run the ingestion pipeline first.")
    db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)

    template = """You are an expert technical assistant for a construction company. Your task is to answer questions strictly based on the technical context provided below.
    Rules:
    1. If the answer is not explicitly in the context, state DIRECTLY: "I cannot find that information in the provided documentation." DO NOT invent or assume anything.
    2. Maintain a technical, precise, and professional tone.
    3. Always cite the source document and page number at the end of your answer if available in the context in a clean format like:
    **Sources:**
     - Folder: [folder name], Document: [document name], Page: [page number] ¨
    Context: {context}
    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: list[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain