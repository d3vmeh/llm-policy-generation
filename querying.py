from typing import Tuple, List, Optional
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
import pdb
import os


from wordcloud import WordCloud
import matplotlib.pyplot as plt

from create_communities import get_community_id, load_summaries

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

print("Running querying.py")
summaries = load_summaries()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
print("Loading Graph")
graph = Neo4jGraph()
print("Graph loaded") 
wordcloud = None

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid", #searching embedding and keyword
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, law, technology, policy, country, organization, or business entities that "
        "appear in the text",
    )   

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    print("RUNNING STRUCTURED RETRIEVER")
    result = ""
    nodes= []
    neighbors = []
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print(entities.names)
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output, node.id AS nodeId, neighbor.id AS neighborId
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output, node.id AS nodeId, neighbor.id AS neighborId
            }
            RETURN output, nodeId, neighborId LIMIT 100
            """,
            {"query": generate_full_text_query(entity)},
        )
        

        #Third relationship
        alternate_query = """CALL db.index.fulltext.queryNodes('entity', $query, {limit:3})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)-[r2:!MENTIONS]->(neighbor2)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id + ' - ' + type(r2) + '->' + neighbor2.id AS output, node.id AS nodeId, neighbor.id AS neighborId, neighbor2.id AS neighbor2Id
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)<-[r2:!MENTIONS]-(neighbor2)
              RETURN neighbor2.id + ' - ' + type(r2) + '->' + neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output, node.id AS nodeId, neighbor.id AS neighborId, neighbor2.id AS neighbor2Id
            }
            RETURN output, nodeId, neighborId LIMIT 100
            """



        for n in response:
            node = n['nodeId']
            if node not in nodes:
                nodes.append(node)        
        for e in response:
            neighbor = e['neighborId']
            if neighbor not in neighbors:
                neighbors.append(neighbor)

        result += "\n".join([el['output'] for el in response])
    #print("Nodes:",nodes)
    #print("Neighbors:",neighbors)
    #print("Returning:",nodes,neighbors)
    return result, nodes, neighbors

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data, related_nodes, neighbors = structured_retriever(question) #context from graph database - nodes, relationships
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]  #context from graph database - text
    print(structured_data)
    #print(related_nodes)
    #print(neighbors)
    community_ids = []
    for node in related_nodes:
        #print("Node:",node)
        community_ids.append(get_community_id(node))
    for neighbor in neighbors:
        community_ids.append(get_community_id(neighbor))
    
    #print(community_ids)

    s = []
    used = []
    for i in community_ids:
        if i != None and i in summaries.keys() and i not in used:
            #print(summaries[i])
            s.append(summaries[i])
            used.append(i)
            
    print("\nNumber of community summaries used for response:",len(s),"\n")

    summaries_str = "\n\n".join(s)

    final_data = f"""Structured data:
                    {structured_data}

                    Unstructured data:
                    {"#Document ". join(unstructured_data)}

                    Community summaries:
                    {summaries_str}
                    """
    #print(structured_data)
    print("Final data:")
    print(final_data)
    wordcloud = create_wordcloud(final_data)
    return final_data

def create_wordcloud(text):
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    word_cloud.to_file('wordcloud.png')
    return word_cloud


prompt = ChatPromptTemplate.from_messages(
        [
        ("system", """
        You are an experienced advisor and international diplomat assisting the US government in shaping foreign policy. Your role is to provide insightful and comprehensive answers to inquiries by synthesizing structured data, unstructured data, and community summaries. 
        Please respond thoughtfully and thoroughly, 
        ensuring that your answers reflect a deep understanding of global issues and diplomatic nuances.
        """),
        ("user", """
        You are tasked with answering the question based solely on the provided context. Please carefully consider the following sources:                                                                                                               
        
        1. **Structured Data**: This includes major entities and their relationships. Use this information to understand the connections and significance of each entity.

        2. **Unstructured Data**: This contains relevant text extracted from various documents. Analyze this text for pertinent information that supports your response.

        3. **Community Summaries**: Review the summaries that outline communities where closely related entities exist. This context is crucial for a comprehensive understanding of the relationships and influences among the entities.

        Additionally, whenever applicable, cite historical precedents and events to strengthen your answer. Ensure your response is well-supported by the information from the below sources.
        {context}

        Here is the question for you to answer: {question}
        Use natural language and be detailed and thorough.
        Answer:
        """)
        ]
        )

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)
#llm = Ollama(model="llama3.2",temperature=0.6)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Finished querying.py")


