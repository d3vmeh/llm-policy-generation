from typing import Tuple, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

import os
from langchain_core.documents import Document
from timebudget import timebudget
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader


def load_documents():
    print("Loading text documents")
    filenames = os.listdir("text_documents/")
    documents = []
    for f in filenames:
        if not f.endswith(".txt"):
            continue
        path = os.path.join("text_documents/", f)
        f = open(path, "r", encoding="utf-8")
        text = f.read()
        document = Document(text)
        documents.append(document)
    print(len(documents),"text documents loaded")

    print("Loading PDFs")
    doc_loader = PyPDFDirectoryLoader("PDFs/")
    docs = doc_loader.load()
    documents += docs
    print("Loaded documents")
    return documents


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


graph = Neo4jGraph()
raw_documents = load_documents()


text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents)
print("Number of documents:", len(documents))
print(documents[0])



llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Using Neo4j LLMGraphTransformer to convert the documents
llm_transformer = LLMGraphTransformer(llm=llm)

print("Converting to graph documents")
with timebudget("Time to convert documents: "):  
    graph_documents = llm_transformer.convert_to_graph_documents(documents)


graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)
print("Documents created")


vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid", #search being performed on embedding and keyword as well
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
graph._driver.close()