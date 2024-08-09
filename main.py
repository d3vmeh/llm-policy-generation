from knowledge_graph import *


documents = load_documents(query = [
    "Cold War",
    "Cold War in Asia",
    "Outline of the Cold War",
    "31st Transportation Battalion",
    "1967 South Vietnam Independence Cup",
    "1979 Salvadoran coup d'état",
    "CIA and the Cultural Cold War",
    ])

print("# of Documents:", len(documents))
print(documents[0])

graph = convert_documents(documents)
create_vector_index()

chain = create_graph(graph, documents)


while True:
    q = input("Enter a query: ")
    print("\n")
    print(structured_retriever(q, graph))
    print(chain.invoke({"question": q}))
    print("\n\n")

