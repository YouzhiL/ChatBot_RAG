from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_parse import LlamaParse
from dotenv import load_dotenv

def split_to_nodes(documents):
  ## Parse to Nodes
  llm = OpenAI(model="gpt-3.5-turbo-0125")
  node_parser = MarkdownElementNodeParser(llm=llm,num_workers=8)
  nodes = node_parser.get_nodes_from_documents(documents)
  base_nodes, index_nodes = node_parser.get_nodes_and_objects(nodes)
  return base_nodes, index_nodes