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

from create_communities import get_community_id, load_summaries

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


summaries = load_summaries()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
graph = Neo4jGraph()


llm = Ollama(model_name="llama3",temperature=0.5)

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
        description="All the person, organization, or business entities that "
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
            RETURN output, nodeId, neighborId LIMIT 200
            """,
            {"query": generate_full_text_query(entity)},
        )
        
        for n in response:
            node = n['nodeId']
            if node not in nodes:
                nodes.append(node)
        

        
        for e in response:
            neighbor = e['neighborId']
            if neighbor not in neighbors:
                neighbors.append(neighbor)
        

        

        result += "\n".join([el['output'] for el in response])
    print("Nodes:",nodes)
    print("Neighbors:",neighbors)
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
    #print("Final data:")
    #print(final_data)
    return final_data

prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are an experienced advisor and international diplomat who is assisting the US government in foreign policy. You use natural language "
         "to answer questions based on structured data, unstructured data, and community summaries. You are thoughtful and thorough in your responses."),
        ("user", """
        Answer the question based only on the following context. The structured data shows major entities and their relationships which you should consider 
         in your response. The unstructured data shows the relevant text from the 
         documents which you should also consider when preparing your response.
         The community summaries show the summary of communities in which closely related entities are present, which
         you should also consider in your response. Also, cite historical precedents and events in support of your answer
         whenever they are relevant:
        {context}

        Question: {question}
        Use natural language and be detailed and thorough.
        Answer:
        """)
        ]
        )

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
llm = Ollama(model_name="llama3",temperature=0.7)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

