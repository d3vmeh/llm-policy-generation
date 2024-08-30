from querying import *
import sys
import streamlit as st

# print("Loaded all")
# i = 0
# question = st.text_input("Enter a prompt",key=str(i))
# is_clicked = st.button("Submit")

# while True:
        
    
#     # print(type(question))
#     # if question.lower() == "exit" or question.lower() == "q":
#     #     break
#     if is_clicked:
#         response = chain.invoke(question)
#         print(type(response))
#         st.write(response)
#         i += 1
#         is_clicked = False
while True:
    q = input("\nEnter a query: ")
    if q.lower() == "exit" or q.lower() == "q":
        break
    print("\n")
    #print("Structured Data:")
    #print(structured_retriever(q))
    print("\n")

    #Unstructured Data
    #print([el.page_content for el in vector_index.similarity_search(q)])

    response = chain.invoke(q)
    print(response)
    print("\n\n")
    f = open("policy_drafts.txt",'a')
    f.write("\n\nQuestion: " + q + "\n")
    f.write("Response: \n" + response + "\n")
    f.write("=============================================")
    f.close()
    
