
import os
from typing import List, Tuple, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama

#This will make it way easier to create graphs using the LLM
from langchain_experimental.graph_transformers import LLMGraphTransformer


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import Neo4jVector, remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
import json
import pickle

import time


"""
Inspired by 'Realtime Powerful RAG Pipeline using Neo4j(Knowledge Graph Db) and Langchain' by https://github.com/sunnysavita10
"""

class Entities(BaseModel):
    #Used to identify information about entitites

    names: List[str] = Field(..., description="All the person, organization, or business entities that "
                             "appear in the text")



def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()




# Fulltext index query
def structured_retriever(question: str, graph) -> str:
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
    structured_data = structured_retriever(question, graph)
    unstructured_data = [el.page_content for el in vector_index2.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")
#llm = Ollama(temperature = 0, model = "llama3", base_url = "http://192.168.2.30:11434")

llm_transformer = LLMGraphTransformer(llm=llm)



NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
AURA_INSTANCEID = os.environ["AURA_INSTANCEID"]
AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]


def load_documents(query = [""], num_docs = None):

    print("Loading documents from Wikipedia")
    raw_documents = []
    for q in query:
        raw_data = WikipediaLoader(query = q).load()
        raw_documents.append(raw_data)
        print("Loaded:",q)
        time.sleep(1)
    
    print('Documents Loaded')
    print(type(raw_documents))
    #Need to split the document into chunks
    print("Splitting documents into chunks")
    text_splitter = TokenTextSplitter(chunk_size = 512, chunk_overlap = 24)
    if num_docs is not None:
        raw_documents = raw_documents[:num_docs]
    print(len(raw_documents))
    documents = []
    for doc in raw_documents:
        doc = text_splitter.split_documents(doc)
        documents += doc
    #documents = text_splitter.split_documents(raw_documents) #only collecting the first three documents
    print("Documents Split")
    return documents

def convert_documents(documents, path = None):
    global graph
    graph = Neo4jGraph()

    print("Converting documents to graph documents")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print("Documents converted")


    # # Save the JSON data to a file
    # with open('graph_data.json', 'w') as file:
    #     file.write(graph_data_json)

    # if path != None:
    #     try:
    #         with open(path, 'r') as file:
    #             graph_data_json = file.read()
    #         file.close()

    #         # Convert JSON back to the Python data structure
    #         graph_documents = json.loads(graph_data_json)
    #         print("Loaded graph data")
    #     except:
    #         print("Error loading graph data from file")

    #Storing the data in Neo4j graph database. Nodes represent entities, edges represent relationships between entities
    # data is stored as nodes, relationships, and properties (attributes of nodes and relationships like name or age)

    #Cypher Queries are used to create and query the graph
    graph.add_graph_documents( #storing the graph documents in Neo4j database
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    #serialized_graph = serialize_graph(graph)
    
    #pickle.dump(graph, open("graph.pkl", "wb"))
    #export_graph_to_csv(graph,"MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m", r"C:\Users\devxm\Documents\AI Stuff\llm-policy-generation\graphs\exported_graph.csv")
    #graph_json = json.dumps(serialized_graph)
    #with open('graph.json', 'w') as file:
    #    file.write(graph_json)
    #file.close()

    #Directly show the graph resulting from the given cypher query
    default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"
    return graph





def create_vector_index():
#Creating a vector index for the documents that will be used for similarity search
    global vector_index2
    vector_index2 = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        index_name="vector2",
    )

def create_graph(graph, documents):
    

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
    global entity_chain
    entity_chain = prompt | llm.with_structured_output(Entities)



    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)



    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language in your answer.
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    global chain
    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

