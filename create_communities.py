from graphdatascience import GraphDataScience
from graphdatascience.server_version.server_version import ServerVersion
from querying import *
from neo4j import GraphDatabase
import pandas as pd
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import pickle

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
    q = """Put your BRIEF summary below and include a title that is SPECIFIC only to the data in this summary as well. 
            The summary title should not be generic or broad like 'US Foriegn Policy', 
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


#MUST run when updating/resetting the database -- also requires increasing the Java heap size if using a new DB
#gds.graph.drop("myGraph")

#graph_projection = create_graph_projection()




# graph = Neo4jGraph()
G = gds.graph.get("myGraph")


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

# #Must use gds.util.asNode(nodeId).id to get names. There is no property "name" for the nodes, so gds.util.asNode(nodeId).name returns null
# query = """
#     CALL gds.graph.nodeProperties.stream('myGraph', 'componentId')
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
#     CALL gds.wcc.write('myGraph', { writeProperty: 'community' }) 
#     YIELD nodePropertiesWritten, componentCount;
# """)


# gds.louvain.mutate(G, mutateProperty="community")

# print(gds.graph.nodeProperties.write(G, ["community"]))

# gds.run_cypher(
#     """
#     MATCH (n) WHERE 'louvainCommunityId' IN keys(n) 
#     RETURN n.name, n.louvainCommunityId LIMIT 10
#     """
# )

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
print("Number of communities:",len(communities.index))
count = 0
for x in range(len(communities.index)):
    c = communities.iloc[x]['comp']
    #c_components = c['comp'].to_list()
    #print(c_components)
    all_community_components.append(c)
    count += 1

print("Count:",count)
#f = open("community_summaries.txt",'w', encoding="utf-8")

summaries = {}
#print(all_community_components)
community_ids = communities['community'].to_list()

# sizes = communities['componentSize'].to_list()

# for s in range(len(sizes)):
#     if 1 < sizes[s] <= 3:
#         print("Community ID:",community_ids[s],"Size:",sizes[s])
#         print(all_community_components[s])
#         print("\n")

# #Uncomment when generating new summaries
# for i in range(len(all_community_components)):
#     #print(i)
#     if sizes[i] > 1:
#         converted_string = ", ".join(str(x) for x in all_community_components[i])

#         #print(converted_string)
#         s = create_community_summary(converted_string)
#         #c = get_community_id(all_community_components[i][0])
#         summaries[community_ids[i]] = s
#         print(s)
#     if sizes[i] <= 1:
#         break

# with open("community_summaries.pkl",'wb') as file:
#     pickle.dump(summaries, file)
#     file.close()

with open('community_summaries.pkl', 'rb') as file: 
      
    # Call load method to deserialze 
    summaries = pickle.load(file) 
  
    print(f"Loaded all summaries. {len(summaries)} from file") 

print(summaries[2247])
#f.close()
#print("Completed")
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

