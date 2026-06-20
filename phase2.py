#Phase1 imports
import streamlit as st

#Phase2 imports
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.title("Rag Chatbot!")

#setup a session state to store the chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

#display the chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Pass your prompt here!")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best,
                                                       the most accurate and the most precise answers. Answer the following question: {user_prompt}.
                                                       Start the answer directly. No small talk please.""")

    model = "llama-3.3-70b-versatile"
    groqchat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name = model) 

    chain = groq_sys_prompt | groqchat | StrOutputParser()
    response = chain.invoke({"user_prompt": prompt})

    #response = "I am your assistant!"  # Placeholder for chatbot response
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})