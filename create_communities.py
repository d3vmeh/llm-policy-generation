from graphdatascience import GraphDataScience
from graphdatascience.server_version.server_version import ServerVersion
from querying import *
from neo4j import GraphDatabase
#from data_ingestion import graph



gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

assert gds.server_version() >= ServerVersion(1, 8, 0)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_node_labels_and_relationship_types(tx):
    # Query to get node labels
    node_labels_query = "CALL db.labels()"
    node_labels_result = tx.run(node_labels_query)
    node_labels = [record["label"] for record in node_labels_result]

    # Query to get relationship types
    relationship_types_query = "CALL db.relationshipTypes()"
    relationship_types_result = tx.run(relationship_types_query)
    relationship_types = [record["relationshipType"] for record in relationship_types_result]

    return node_labels, relationship_types

with driver.session() as session:
    node_labels, relationship_types = session.read_transaction(get_node_labels_and_relationship_types)

# Convert node labels and relationship types to the format expected by gds.graph.project
node_projection = {label: {} for label in node_labels}  # Assuming no properties to include
relationship_projection = {rel_type: {} for rel_type in relationship_types}  # Assuming no properties to include

print("Starting projection")
# Project the graph using the extracted schema




# graph_projection = gds.graph.project(
#     "myGraph",
#     node_projection,
#     relationship_projection
# )
# print("=============================================")

# print(type(graph_projection))

# print(graph.get_structured_schema)
# graph_projection = gds.graph.project(
#     "myGraph",
#     {
#         "NodeLabel": {}  # Specify node labels and properties here
#     },
#     {
#         "REL_TYPE": {}  # Specify relationship types and properties here
#     }
# )

# # result = gds.wcc.mutate(graph, mutateProperty = "componentId")
# # print("Components found:", result.componentCount)

graph = Neo4jGraph()

#G, result = gds.graph.project("testingGraph",)



G, r = gds.graph.get("")

#print(G.node_labels)
gds.louvain.mutate(G, mutateProperty="louvainCommunityId")


print(gds.graph.nodeProperties.write(G, ["louvainCommunityId"]))



gds.run_cypher(
    """
    MATCH (n) WHERE 'louvainCommunityId' IN keys(n) 
    RETURN n.name, n.louvainCommunityId LIMIT 10
    """
)
