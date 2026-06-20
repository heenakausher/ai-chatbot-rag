import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain)
from langchain_classic.chains.retrieval import (
    create_retrieval_chain)


# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()


# Streamlit Page Config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄"
)

st.title("📄 PDF RAG Chatbot")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Create Vector Store

@st.cache_resource
def get_vectorstore():

    pdf_path = "Heena Kausher_CV.PDF"  # Replace with your PDF file

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    loader = PDFPlumberLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# Create LLM

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )


# Prompt Template

prompt_template = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question only from the provided context.

If the answer is not found in the context,
respond with:

"I could not find that information in the document."

Context:
{context}

Question:
{input}

Answer:
"""
)


# Create RAG Chain

@st.cache_resource
def get_rag_chain():

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20
        }
    )

    llm = get_llm()

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return retrieval_chain


# Chat Input

user_question = st.chat_input(
    "Ask a question about the PDF..."
)

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    try:

        rag_chain = get_rag_chain()

        response = rag_chain.invoke(
            {
                "input": user_question
            }
        )

        answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")