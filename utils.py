from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
# from llama_index.core import Settings
# from llama_index.core.node_parser import MarkdownElementNodeParser
# from llama_index.core.ingestion import IngestionCache, IngestionPipeline
# from llama_parse import LlamaParse
# from page_nodes_generator import get_page_nodes
# from global_setting import CACHE_FILE

def get_doc_tools(file_path: str, name:str) -> str:
  """Get vector query and summary query tools from a document."""
  
  # Load documents
  documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
  splitter = SentenceSplitter(chunk_size=1024)
  nodes = splitter.get_nodes_from_documents(documents)
  vector_index = VectorStoreIndex(nodes)
  """Use to answer questions over a given paper.

    Useful if you have specific questions over the paper.
    Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    """
    
  def vector_query(
    query: str,
    page_numbers:Optional[List[str]] = None
  ) -> str:
    """Use to answer questions over a given paper.

    Useful if you have specific questions over the paper.
    Always leave page_numbers as None UNLESS there is a specific page you want to search for.

    Args:
        query (str): the string query to be embedded.
        page_numbers (Optional[List[str]], optional): Filter by set of pages. Defaults to None.
    """
    page_numbers = page_numbers or []
    metadata_dicts = [
      {"key": "page_label","value":p} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
      similarity_top_k = 2,
      filters=MetadataFilters.from_dicts(
        metadata_dicts,
        condition=FilterCondition.OR
      )
    )
    response = query_engine.query(query)
    return response
  
  vector_query_tool = SummaryTool.from_defaults(
    name=f"vector_tool{name}",
    fn=vector_query
  )
  
  summary_index = SummaryIndex(nodes)
  summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
  )
  summary_tool = QueryEngineTool.from_defaults(
    name=f"summary_tool_{name}",
    query_engine=summary_query_engine,
    description=(
      f"Useful for summarization questions related to {name}"
    )
  )
  
  return vector_query_tool, summary_tool

llm = OpenAI()
vector_query_tool, summary_tool = get_doc_tools(file_path, name)
respose = llm.predict_and_call(
  [vector_query_tool, summary_tool],
  "What is ... on page 6?",
  verbose=True
)
