import streamlit as st
import requests
import json
import uuid

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
            "http://localhost:5000/chat",
            json={"message": prompt, "session_id": st.session_state.session_id},
            stream=True,
        ).iter_lines():
            if response:
                data = json.loads(response.decode())
                if data["type"] == "content":
                    full_response += data["content"]
                    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())