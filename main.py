from querying import *


while True:
    q = input("Enter a query: ")
    print("\n")
    print(structured_retriever(q))
    print("\n")
    print([el.page_content for el in vector_index.similarity_search(q)])
    print(chain.invoke(q))
    print("\n\n")

