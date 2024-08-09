from graphdatascience import GraphDataScience
from graphdatascience.server_version.server_version import ServerVersion
from neo4j import GraphDatabase

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import pickle
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


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

def create_graph_projection(graphName="myGraph0"):
    print("Creating graph projection")
    graph_projection = gds.graph.project(
    graphName,
    node_projection,
    relationship_projection,
    )


    print("Graph projection created")
    print("=============================================")






    """If getting not enough heap space error, run this in neo4j. It will generate
    a graph named myGraph0. Set the total nodes at the top and make sure it is equal
    to the batch size to avoid generating subgraphs:
    
    WITH 12297 AS totalNodes, 12297 AS batchSize
    UNWIND range(0, totalNodes - 1, batchSize) AS batchStart
    CALL {
        WITH batchStart, batchSize
        MATCH (n)
        WHERE id(n) >= batchStart AND id(n) < batchStart + batchSize
        RETURN collect(id(n)) AS batchNodeIds
    }
    WITH batchNodeIds, batchStart
    CALL gds.graph.project.cypher(
        'myGraph' + batchStart,
        'MATCH (n) WHERE id(n) IN $batchNodeIds RETURN id(n) AS id',
        'MATCH (n)-[r]->(m) WHERE id(n) IN $batchNodeIds AND id(m) IN $batchNodeIds RETURN id(n) AS source, id(m) AS target',
        { parameters: { batchNodeIds: batchNodeIds }}
    )
    YIELD graphName AS graph, nodeCount AS nodes, relationshipCount AS rels
    RETURN graph, nodes, rels;
    """



    return graph_projection

def get_local_clustering_coefficients():
    clustering_coefficients = gds.run_cypher("""
        CALL gds.localClusteringCoefficient.stream('myGraph0')
        YIELD nodeId, localClusteringCoefficient
        RETURN gds.util.asNode(nodeId).id AS name, localClusteringCoefficient
        ORDER BY localClusteringCoefficient DESC
    """)
    return clustering_coefficients

def get_node_popularity():
    popularity = gds.run_cypher("""
        CALL gds.degree.stream('myGraph0')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS name, score AS popularity
        ORDER BY popularity DESC            
    """)
    return popularity

def get_triangle_count():
    triangle_count = gds.run_cypher("""
        CALL gds.triangleCount.stream('myGraph0')
        YIELD nodeId, triangleCount
        RETURN gds.util.asNode(nodeId).id AS name, triangleCount
        ORDER BY triangleCount DESC
    """)
    return triangle_count

def create_community_summary(community_components):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are an experienced data analyst who is assisting the US government in consolidating foreign policy data."
         "The data is stored in a list of components that are all related to each other." 
         "You use natural language to summarize the data."), 
        ("user", """
        Only use the following list of components to summarize the data. Use natural language.
        Only include the summary of the data in your response.

        ===============================================================
        Here is the data:

        {components}
        
         
        ===============================================================
        {question}
        
        """
        )
        ]
        )
    
    chain = (
         {"components": lambda x: community_components, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
    )
    q = """Put your detailed and thorough summary below and include a title that is SPECIFIC only to the data in this summary as well. 
            The summary title should not be generic or broad like 'US Foreign Policy' or 'US Relations with China', 
            it should focus on specific details and items mentioned in the data.
            These items can be names of people, countries, concepts, policies, etc..
            Do not just say a broad term such as 'key foreign policy' or 'global relations' without providing more details
            Put your summary and title here:"""
    summary = chain.invoke(q)
    return summary

def get_community_id(node_id: str) -> str:
    response = gds.run_cypher(
        f"""
        MATCH (n) WHERE n.id = "{node_id}"
        RETURN n.community AS communityId
        """
    )
    r = response['communityId'][0]
    print(node_id, r)
    #if response:
    return response['communityId'][0]
    #return 0

def load_summaries():
    with open('community_summaries.pkl', 'rb') as file: 
        # Call load method to deserialze 
        summaries = pickle.load(file) 
    
    print(f"Loaded all summaries. {len(summaries)} from file")
    return summaries


graphName = "myGraph0"

#MUST run when updating/resetting the database -- also requires increasing the Java heap size if using a new DB
#gds.graph.drop("myGraph0")

#graph_projection = create_graph_projection()

# graph = Neo4jGraph()
G = gds.graph.get(graphName)


# print("Searching for weakly connected components")
# result = gds.wcc.mutate(G, mutateProperty = "componentId")
# print("Components found:", result.componentCount)

# print("Searching for strongly connected components")
# result = gds.scc.mutate(G, mutateProperty = "componentId")
# print("Components found:", result.componentCount)




# print("Searching for strongly connected components")
# result = gds.scc.stream(G)
# print(result)
# print("Components found:", result.componentCount)

#Must use gds.util.asNode(nodeId).id to get names. There is no property "name" for the nodes, so gds.util.asNode(nodeId).name returns null
"""
Run to generate communities
"""

# query = """
#     CALL gds.graph.nodeProperties.stream('myGraph0', 'componentId')
#     YIELD nodeId, propertyValue
#     WITH gds.util.asNode(nodeId).id AS node, propertyValue AS componentId
#     WITH componentId, collect(node) AS comp
#     WITH componentId, comp, size(comp) AS componentSize
#     RETURN componentId, componentSize, comp
#     ORDER BY componentSize DESC 
# """
# components = gds.run_cypher(query)
# print(components)

# gds.run_cypher("""
#     CALL gds.wcc.write('myGraph0', { writeProperty: 'community' }) 
#     YIELD nodePropertiesWritten, componentCount;
# """)


# gds.louvain.mutate(G, mutateProperty="community")

print(gds.graph.nodeProperties.write(G, ["community"]))

node_popularity = get_node_popularity()
print(node_popularity,'\n')

community_query = """
    CALL gds.graph.nodeProperties.stream('myGraph0', 'community')
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

all_community_components = []
community_summaries = []

all_community_components = communities['comp'].to_list()
community_ids = communities['community'].to_list()
sizes = communities['componentSize'].to_list()

summaries = {}


total = 0
single_node_communities = 0
double_node_communities = 0
triple_node_communities = 0
quad_node_communities = 0
c = 0
for s in range(len(sizes)):
    if sizes[s] == 1:
        single_node_communities += 1
    else:
        total += 1
        
        if sizes[s] == 2:
            double_node_communities += 1
        if sizes[s] == 3:
            triple_node_communities += 1
        if 4<= sizes[s] < 25:
            quad_node_communities += 1
        else:
            c += 1

print("Single node communities:",single_node_communities)
print("Double node communities:",double_node_communities)
print("Triple node communities:",triple_node_communities)
print("Quad node communities:",quad_node_communities)
print("Communities between 4 and 25 nodes:",c)
print(f"There will be {total} community summaries generated")

"""
Uncomment when generating new community summaries
"""


# count = 0
# for i in range(len(all_community_components)):
#     #print(i)
#     if sizes[i] > 1:
#         converted_string = ", ".join(str(x) for x in all_community_components[i])

#         #print(converted_string)
#         s = create_community_summary(converted_string)
#         #c = get_community_id(all_community_components[i][0])
#         summaries[community_ids[i]] = s
#         print(s)
#         print(f"\n{count}/{total} community summaries generated\n")
#         count += 1
#     if sizes[i] <= 1:
#         break

# print(count," community summaries generated")


# with open("community_summaries.pkl",'wb') as file:
#     pickle.dump(summaries, file)
#     file.close()


"""
Loading Summaries
"""



# with open('community_summaries.pkl', 'rb') as file: 
      
#     # Call load method to deserialze 
#     summaries = pickle.load(file) 
  
# print(f"Loaded all summaries. {len(summaries)} from file") 


