from graphdatascience import GraphDataScience
from graphdatascience.server_version.server_version import ServerVersion
from neo4j import GraphDatabase

from langchain_community.llms.ollama import Ollama
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


def create_graph_projection():
    # There are currently 31729 nodes in the graph. Unable to run via the Python function due to memory issues
    projection_query ="""
    WITH 31729 AS totalNodes, 31729 AS batchSize
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
        'MATCH (n)-[r]->(m) WHERE id(n) IN $batchNodeIds AND id(m) IN $batchNodeIds RETURN id(n) AS source, id(m) AS target, type(r) AS type',
        { parameters: { batchNodeIds: batchNodeIds }}
    )
    YIELD graphName AS graph, nodeCount AS nodes, relationshipCount AS rels
    RETURN graph, nodes, rels;
    """

    with driver.session() as session:
        result = session.run(projection_query)

    driver.close()
    print("Graph projection created")


# The following three functions are used to show some basic statistics about the graph
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



# Using an LLM to generate a summary of a community given a list of all of its components
def create_community_summary(community_components):


    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    #llm = Ollama(model="llama3.2",temperature=0.5)

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are an experienced data analyst who is assisting the US government in consolidating foreign policy data."
         "The data is stored in a list of components that are all related to each other." 
         "You use natural language to summarize the data."), 
        ("user", """
        Using only the components provided below, create a comprehensive and coherent summary that captures the essence of the data. Ensure your summary includes a title that is specific to the content, focusing on distinct details such as names, countries, concepts, policies, or significant events mentioned in the data.

        ===============================================================
        Here is the data:

        {components}
        ===============================================================

        **Title:** [Insert a specific title related to the data]

        **Summary:**
        In your summary, avoid generic terms like 'foreign policy' or 'global relations.' Instead, delve into the particulars of the components, clearly articulating their significance and interconnections. Your summary should be thorough, easy to understand, and devoid of bullet points, ensuring that all important components are mentioned.

        {question}
        Please provide your detailed summary and title below:
        
        
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

    # Community summary generation prompt
    q = """Put your detailed and thorough summary below and include a title that is SPECIFIC only to the data in this summary as well. 
            
            
            
            The summary title should not be generic or broad like 'US Foreign Policy' or 'US Relations with China'. Do not use broad
            term like 'foreign policy' or 'global relations'. The title should be specific to the data in the summary and
            it should focus on specific details and items mentioned in the data.
            These items can be names of people, countries, concepts, policies, etc..


            Do not just say a broad term such as 'key foreign policy' or 'global relations' without providing more details.
            Do not use bullet points.


            You must mention all of the important components of the community in the summary. Make sure the summary is clear, easy to understand, and easy to analyze.

            Put your summary and title here:"""
    

    q = ""
    
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



driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def create_communities_in_graph():
    #Must use gds.util.asNode(nodeId).id to get names. There is no property "name" for the nodes, so gds.util.asNode(nodeId).name returns null
   
    # Adds componentId property to nodes as well
    print("Searching for weakly connected components")
    result = gds.wcc.mutate(G, mutateProperty = "componentId")
    print("Components found:", result.componentCount)

    query = """
        CALL gds.graph.nodeProperties.stream('myGraph0', 'componentId')
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
        CALL gds.wcc.write('myGraph0', { writeProperty: 'community' }) 
        YIELD nodePropertiesWritten, componentCount;
    """)

    gds.leiden.mutate(G, mutateProperty="community")
    # gds.louvain.mutate(G, mutateProperty="community")

    print(gds.graph.nodeProperties.write(G, ["community"]))

    print(get_node_popularity(),'\n')
    print(get_local_clustering_coefficients(),'\n')
    print(get_triangle_count(),'\n')



def create_undirected_relationships():

    create_undirected_relationships_query = """
    
    CALL db.relationshipTypes() YIELD relationshipType
    WITH collect(relationshipType) AS relationshipTypes
    UNWIND relationshipTypes AS rType
    CALL gds.graph.relationships.toUndirected(
    'myGraph0',
    {relationshipType: rType, mutateRelationshipType: rType + '_UNDIRECTED'}
    )
    YIELD inputRelationships, relationshipsWritten
    RETURN rType, inputRelationships, relationshipsWritten
    """
    with driver.session() as session:
        result = session.run(create_undirected_relationships_query)
        #for record in result:
        #    print(f"Converted {record['inputRelationships']} relationships of type {record['rType']} to undirected relationships")
    print("Undirected relationships created")



#Drop non-undirected relationships
def drop_relationship_types():
    with driver.session() as session:
        result = session.run("""
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) AS allRelTypes
        """)
        all_rel_types = result.single()["allRelTypes"]
        
        #Filter out '_UNDIRECTED' and drop relationships
        for r_type in all_rel_types:
            if not r_type.endswith('_UNDIRECTED'):
                session.run("""
                    CALL gds.graph.relationships.drop(
                        'myGraph0',
                        $relationshipType
                    )
                """, parameters={"relationshipType": r_type})
                print(f"Dropped relationship type: {r_type}")
    driver.close()

def print_relationship_types():
    with driver.session() as session:
        # Query to get all relationship types in the graph projection
        query = """
        CALL gds.graph.list() YIELD graphName
        WHERE graphName = 'myGraph0'
        CALL gds.graph.relationships.stream(graphName)
        YIELD relationshipType
        RETURN DISTINCT relationshipType
        """
        result = session.run(query)
        
        count = 0
        for record in result:
            #print(f"Relationship Type: {record['relationshipType']}")
            count += 1
        print(count, "relationship types found")
    driver.close()


"""
Uncomment to create graph projection and undirected relationships
"""

#create_graph_projection()


#create_undirected_relationships()
#drop_relationship_types()

graphName = "myGraph0"


"""
Need to run if creating a new graph projection
"""
# gds.graph.drop("myGraph0")


G = gds.graph.get(graphName)

"""
Run to generate communities
"""
#create_communities_in_graph()


"""
For getting some info about community size, components, etc.
"""

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
medium_communities = 0
c = 0
for s in range(len(sizes)):

    #Ignoring single node communities
    if sizes[s] == 1:
        single_node_communities += 1
    else:
        total += 1
        
        if sizes[s] == 2:
            double_node_communities += 1
        if sizes[s] == 3:
            triple_node_communities += 1
        if sizes[s] == 4:
            quad_node_communities += 1
        if 4<= sizes[s] < 25:
            medium_communities += 1

print("Single node communities:",single_node_communities)
print("Double node communities:",double_node_communities)
print("Triple node communities:",triple_node_communities)
print("Quad node communities:",quad_node_communities)
print("Communities between 4 and 25 nodes:",medium_communities)
print(f"There will be {total} community summaries")

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

with open('community_summaries.pkl', 'rb') as file: 
    summaries = pickle.load(file) 
  
print(f"Loaded all summaries. {len(summaries)} from file") 


