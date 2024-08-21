from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_parse import LlamaParse
from dotenv import load_dotenv

from page_nodes_generator import get_page_nodes
from document_loader import load_documents
from md_node_parser import split_to_nodes

def build_vec_index(base_nodes, index_nodes, page_nodes):
  # recursive_index = VectorStoreIndex(nodes=base_nodes+index_nodes+page_nodes)
  recursive_index = VectorStoreIndex(nodes=base_nodes)
  recursive_query_engine = recursive_index.as_query_engine(verbose=True)
  return recursive_query_engine