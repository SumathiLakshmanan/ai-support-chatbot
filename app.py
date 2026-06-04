import streamlit as st
import uuid
from chatbot import get_chatbot, ask

# Page config
st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="centered"
)

# Custom styling

st.title("🤖 AI Customer Support")
st.caption("Powered by AWS Bedrock + LangChain + Pinecone")

# Initialize chatbot once
if "chatbot" not in st.session_state:
    with st.spinner("Loading AI..."):
        st.session_state.chatbot = get_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm your AI support assistant. How can I help you today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(st.session_state.chatbot, question)
            st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })