from querying import *
import sys
import streamlit as st

print("Loaded all")



print("Ready to answer questions")

st.title("AI Foreign Policy Assistant")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

user_query = st.chat_input("Enter a question")

if user_query != None and user_query != "":

    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        #context = query_database(user_query, db)
        #response = get_response(context, user_query, llm)
        response = chain.invoke(user_query)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(response))




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
    
