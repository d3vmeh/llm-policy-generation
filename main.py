from querying import *
import sys
import streamlit as st
import shelve
from PIL import Image


def load_chat_history():
    db = shelve.open("conversation_history")
    return db.get("messages",[])
    
def save_chat_history(messages):
    db = shelve.open("conversation_history")
    db["messages"] = messages

print("Loaded all")
print("Ready to answer questions")

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
        
        #print(context)
        response = chain.invoke(query)
        message_placeholder.markdown(response) 
        try:
            img = Image.open('wordcloud.png')
            st.image(img, caption='Wordcloud of the response', use_column_width=True)
        except:
            pass  
    st.session_state.messages.append({"role": "assistant", "content": response})

save_chat_history(st.session_state.messages)








# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# chat_placeholder = st.container()
# prompt_placeholder = st.form("chat-form")

# user_query = st.chat_input("Enter a question")

# if user_query != None and user_query != "":

#     st.session_state.chat_history.append(HumanMessage(user_query))

#     with st.chat_message("Human"):
#         st.markdown(user_query)

#     with st.chat_message("AI"):
#         #context = query_database(user_query, db)
#         #response = get_response(context, user_query, llm)
#         response = chain.invoke(user_query)
#         st.markdown(response)

#     st.session_state.chat_history.append(AIMessage(response))




# i = 0
# question = st.text_area("Enter a prompt",key=str(i))
# is_clicked = st.button("Submit")


# if is_clicked:
#     print("clicked")
#     st.write("Response:")
#     response = chain.invoke(question)
#     print(type(response))
#     st.write(response)
#     st.image("wordcloud.png")
#     i += 1
#     #is_clicked = False
# while True:
#     q = input("\nEnter a query: ")
#     if q.lower() == "exit" or q.lower() == "q":
#         break
#     print("\n")
#     #print("Structured Data:")
#     #print(structured_retriever(q))
#     print("\n")

#     #Unstructured Data
#     #print([el.page_content for el in vector_index.similarity_search(q)])

#     response = chain.invoke(q)
#     print(response)
#     print("\n\n")
#     f = open("policy_drafts.txt",'a')
#     f.write("\n\nQuestion: " + q + "\n")
#     f.write("Response: \n" + response + "\n")
#     f.write("=============================================")
#     f.close()
    
