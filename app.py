import streamlit as st
from medical_rag import ask_medical_question, load_rag_system

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Medical AI Assistant", layout="centered")

st.title("🩺 Medical Knowledge Assistant")
st.write("Ask medical questions based on trusted documents")

# -------------------------------
# LOAD SYSTEM (CACHE)
# -------------------------------
@st.cache_resource
def load_system():
    return load_rag_system()

retriever, llm = load_system()

# -------------------------------
# Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# User Input
# -------------------------------
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_medical_question(user_input, retriever, llm)
            st.markdown(response)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": response})