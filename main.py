from knowledge_graph import *


documents = load_documents(query = [
    "Cold War",
    "Containment",
    "Deterrence theory",
    "Domino theory",
    "Flexible response",
    "Lacy-Zarubin Agreement",
    "Linkage (policy)",
     "Massive retaliation",
    "Mutual assured destruction",
    "New Look (policy)",
    "Paasikivi-Kekkonen doctrine",
    "Reverse Course",
    "Rollback",
    "Schlesinger Doctrine"
    ])

print("# of Documents:", len(documents))
print(documents[0])

graph = convert_documents(documents, "graph_data.json")
create_vector_index()

print("Creating graph")
chain = create_graph(graph, documents)
print("Graph created")


while True:
    q = input("Enter a query: ")
    print("\n")
    print(structured_retriever(q, graph))
    print("\n")
    print(chain.invoke({"question": q}))
    print("\n\n")

