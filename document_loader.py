from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_parse import LlamaParse
from page_nodes_generator import get_page_nodes
from global_setting import CACHE_FILE

def load_documents(file_path):
  ## Load documents
  doc_parser = LlamaParse(result_type="markdown", verbose=True)
  documents = doc_parser.load_data("data/n_beats.pdf")
  # contents = []
  # for doc in documents:
  #   contents.append(doc.get_content())
  return documents
  