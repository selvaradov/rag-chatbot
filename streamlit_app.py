import streamlit as st
import requests
import json
import uuid
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Get backend
BACKEND_URL = os.environ.get("BACKEND_URL")
if not BACKEND_URL:
    st.error("BACKEND_URL is not set in the environment variables.")
    st.stop()

# Set up the Streamlit app
st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:")
st.title("AI Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to your backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": prompt, "session_id": st.session_state.session_id},
            stream=True,
        ).iter_lines():
            if response:
                data = json.loads(response.decode())
                if data["type"] == "content":
                    content = data["content"]
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "type" in item:
                                if item["type"] == "text":
                                    full_response += item.get("text", "")
                            elif isinstance(item, str):
                                full_response += item
                    elif isinstance(content, str):
                        full_response += content
                    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()
