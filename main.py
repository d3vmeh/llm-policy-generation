from querying import *



while True:
    q = input("\nEnter a query: ")
    print("\n")
    print(structured_retriever(q))
    print("\n")
    #print([el.page_content for el in vector_index.similarity_search(q)])

    response = chain.invoke(q)
    print(response)
    print("\n\n")
    f = open("policy_drafts.txt",'a')
    f.write("\n\nQuestion: " + q + "\n")
    f.write("Response: \n" + response + "\n")
    f.write("=============================================")
    f.close()