# ============================================
# app.py - Streamlit Web Interface
# ============================================

import streamlit as st
from chatbot import create_chatbot, get_answer

# -----------------------------------------------
# Step 1: Page settings
# -----------------------------------------------
st.set_page_config(
    page_title="Customer Support Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Customer Support Chatbot")
st.caption("Ask me anything about our company!")

# -----------------------------------------------
# Step 2: Load chatbot only once
# -----------------------------------------------
# st.session_state saves data between reruns
# Without this, chatbot reloads every time user types

if "chain" not in st.session_state:
    with st.spinner("Loading chatbot..."):
        st.session_state.chain = create_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history for display

# -----------------------------------------------
# Step 3: Display previous chat messages
# -----------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # "user" or "assistant"
        st.write(message["content"])

# -----------------------------------------------
# Step 4: Handle new user input
# -----------------------------------------------
if prompt := st.chat_input("Type your question here..."):

    # Show user message
    with st.chat_message("user"):
        st.write(prompt)

    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get answer from chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(st.session_state.chain, prompt)
        st.write(answer)

    # Save assistant answer to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
