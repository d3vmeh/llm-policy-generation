from querying import *
import sys


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
    
