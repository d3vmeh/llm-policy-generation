from knowledge_graph import *
from database_tools import *

documents = load_documents(query = [
    "China–United States relations",
    "History of China–United States relations",
    #"The Coming Conflict with China",
    # "Restrictions on TikTok in the United States",
    # "State visit by Xi Jinping to the United States",
    # "Stealth War",
    # "SARS conspiracy theory",
    # "Stealth War",
    # "Taiwan Allies International Protection and Enhancement Initiative Act",
    # "Taiwan Relations Act",
    # "United States Department of Defense China Task Force",
    # "Treaty of Friendship, Commerce and Navigation between the United States of America and the Republic of China",
    ])

print("# of Documents:", len(documents))
print(documents[0])

graph = convert_documents(documents[:3], "graph_data.json")


#Save graph:

exporter = GraphExporter("neo4j://localhost:7687", "neo4j", NEO4J_PASSWORD)
exporter.export_to_csv("nodes.csv", "llm-policy-generation/graphs/relationships.csv")
exporter.close()
print("Graph exported to CSV")

# Example usage
importer = GraphImporter("neo4j://localhost:7687", "neo4j", NEO4J_PASSWORD)
importer.import_from_csv("nodes.csv", "llm-policy-generation/graphs/relationships.csv")
importer.close()
print("Graph imported from CSV")



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

