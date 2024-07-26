# Extract entities from text
from typing import Tuple, List, Optional

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

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

#breakpoint()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
graph = Neo4jGraph()



vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid", #search being performed on embedding and keyword as well
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
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question) #context from graph database - nodes, relationships
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]  #context from graph database - text
    final_data = f"""Structured data:
                    {structured_data}

                    Unstructured data:
                    {"#Document ". join(unstructured_data)}
                    """
    return final_data

prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are an experienced advisor and international diplomat who is assisting the US government in foreign policy. You use natural language "
         "to answer questions based on structured and unstructured data. You are thoughtful and thorough in your responses."),
        ("user", """
        Answer the question based only on the following context. The structured data shows major entities and their relationships which you should consider 
         in your respons. The unstructured data shows the relevant text from the 
         documents which you should also consider when preparing your response:
        {context}

        Question: {question}
        Use natural language and be detailed and thorough.
        Answer:
        """)
        ]
        )

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#Testing
# breakpoint()
# question = "Which house did Elizabeth I belong to?"
# structured_data = structured_retriever(question)
# print(structured_data)
# unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
# print(unstructured_data)
# print(retriever(question))

# breakpoint()
# resp = chain.invoke("Which house did Elizabeth I belong to?")
# print(resp)

# resp = chain.invoke("What were Elizabeth's key achievements?")
# print(resp)

# resp = chain.invoke("Write a 500 word paragraph on Elizabeth's life, highlighting her key achievements?")
# print(resp)

# graph._driver.close()
