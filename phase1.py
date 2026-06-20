#Phase1 imports
import streamlit as st

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
    response = "I am your assistant!"  # Placeholder for chatbot response
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})