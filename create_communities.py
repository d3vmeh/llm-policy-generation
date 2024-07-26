from graphdatascience import GraphDataScience
from graphdatascience.server_version.server_version import ServerVersion
from querying import *
from neo4j import GraphDatabase
import pandas as pd


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
relationship_projection = {rel_type: {'orientation':'UNDIRECTED'} for rel_type in relationship_types}  # Assuming no properties to include

def create_graph_projection():
    print("Creating graph projection")
    graph_projection = gds.graph.project(
    "myGraph",
    node_projection,
    relationship_projection,
    )
    print("Graph projection created")
    print("=============================================")
    return graph_projection

def get_local_clustering_coefficients():
    clustering_coefficients = gds.run_cypher("""
        CALL gds.localClusteringCoefficient.stream('myGraph')
        YIELD nodeId, localClusteringCoefficient
        RETURN gds.util.asNode(nodeId).id AS name, localClusteringCoefficient
        ORDER BY localClusteringCoefficient DESC
    """)
    return clustering_coefficients

def get_node_popularity():
    popularity = gds.run_cypher("""
        CALL gds.degree.stream('myGraph')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS name, score AS popularity
        ORDER BY popularity DESC            
    """)
    return popularity

def get_triangle_count():
    triangle_count = gds.run_cypher("""
        CALL gds.triangleCount.stream('myGraph')
        YIELD nodeId, triangleCount
        RETURN gds.util.asNode(nodeId).id AS name, triangleCount
        ORDER BY triangleCount DESC
    """)
    return triangle_count


#MUST run when updating/resetting the database -- also requires increasing the Java heap size if using a new DB
gds.graph.drop("myGraph")

graph_projection = create_graph_projection()




graph = Neo4jGraph()
G = gds.graph.get("myGraph")


print("Searching for weakly connected components")
result = gds.wcc.mutate(G, mutateProperty = "componentId")
print("Components found:", result.componentCount)


#Must use gds.util.asNode(nodeId).id to get names. There is no property "name" for the nodes, so gds.util.asNode(nodeId).name returns null
query = """
    CALL gds.graph.nodeProperties.stream('myGraph', 'componentId')
    YIELD nodeId, propertyValue
    WITH gds.util.asNode(nodeId).id AS node, propertyValue AS componentId
    WITH componentId, collect(node) AS comp
    WITH componentId, comp, size(comp) AS componentSize
    RETURN componentId, componentSize, comp
    ORDER BY componentSize DESC 
"""
components = gds.run_cypher(query)
print(components)

gds.run_cypher("""
    CALL gds.wcc.write('myGraph', { writeProperty: 'community' }) 
    YIELD nodePropertiesWritten, componentCount;
""")


gds.louvain.mutate(G, mutateProperty="community")

print(gds.graph.nodeProperties.write(G, ["community"]))

gds.run_cypher(
    """
    MATCH (n) WHERE 'louvainCommunityId' IN keys(n) 
    RETURN n.name, n.louvainCommunityId LIMIT 10
    """
)

clustering_coefficients = get_local_clustering_coefficients()
print(clustering_coefficients,'\n')

node_popularity = get_node_popularity()
print(node_popularity,'\n')

triangle_count = get_triangle_count()
print(triangle_count,'\n')

community_query = """
    CALL gds.graph.nodeProperties.stream('myGraph', 'community')
    YIELD nodeId, propertyValue
    WITH gds.util.asNode(nodeId).id AS node, propertyValue AS community
    WITH community, collect(node) AS comp
    WITH community, comp, size(comp) AS componentSize
    RETURN community, componentSize, comp
    ORDER BY componentSize DESC
"""
communities = gds.run_cypher(community_query)
print("\nCommunities:")
print(communities)
#for c in largest_communities:
    #print(c['componentId'])



all_community_components = []
community_summaries = []
for c in communities:
    c_components = communities['comp'].to_list()[0]
    all_community_components.append(c_components)

print(all_community_components[0][:10])


# smallest_community_query = """
#     CALL gds.graph.nodeProperties.stream('myGraph', 'community')
#     YIELD nodeId, propertyValue
#     WITH gds.util.asNode(nodeId).id AS node, propertyValue AS community
#     WITH community, collect(node) AS comp
#     WITH community, comp, size(comp) AS componentSize
#     RETURN community, componentSize, comp
#     ORDER BY componentSize ASC
#     LIMIT 1
# """
# smallest_community = gds.run_cypher(smallest_community_query)
# print("\nSmallest community:")
# print(smallest_community)