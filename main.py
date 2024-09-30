from querying import *
import sys
import streamlit as st
import shelve
from PIL import Image

#Streamlit functions
def load_chat_history():
    db = shelve.open("conversation_history")
    return db.get("messages",[])
    
def save_chat_history(messages):
    db = shelve.open("conversation_history")
    db["messages"] = messages



st.title("AI Foreign Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()



with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = chain.invoke(query)
        message_placeholder.markdown(response) 
        try:
            img = Image.open('wordcloud.png')
            st.image(img, caption='Wordcloud of the response', use_column_width=True)
        except:
            pass  
    st.session_state.messages.append({"role": "assistant", "content": response})

save_chat_history(st.session_state.messages)
