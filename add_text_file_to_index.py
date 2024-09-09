from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, PromptTemplate
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.schema import ImageNode
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core.response.notebook_utils import display_source_node
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test_module.test_settings import QUERIES
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
import psycopg2
from sqlalchemy import make_url
from dotenv import load_dotenv
import vecs
from utils import img_2_b64
from global_setting import TEST_PIPELINE_CACHE

def add_text_to_index(filename):
  ### Create vector store
  load_dotenv()
  DB_PASSWORD =  os.getenv("DB_PASSWORD")
  connection_string = f"postgresql://postgres.mdhkfgcsxrasokapahiv:{DB_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
  # image_collection = vecs.get_or_create_collection(name="image_collection_dev", dimension=512)
  # text_collection = vecs.get_or_create_collection(name="text_collection_dev")
  vector_store = SupabaseVectorStore(postgres_connection_string=connection_string, collection_name="text_collection_test")
  image_store = SupabaseVectorStore(postgres_connection_string=connection_string, collection_name="image_collection_test", dimension=512)
  storage_context = StorageContext.from_defaults(vector_store=vector_store, image_store=image_store)

  ### Load Documents
  # filename = "data/combined_test/1-feasibility_report.pdf"
  raw_documents = SimpleDirectoryReader(input_files=[filename], filename_as_id=True).load_data()
  ### Add doc metadata
  """An example of doc metadata:
  file_path: /Users/youzhi/workspace/cronwell/ChatBot_RAG/data/combined_test/01.png
  file_name: 01.png
  file_type: image/png
  file_size: 169895
  creation_date: 2024-08-29
  last_modified_date: 2024-08-29
  user: Judy
  project: test_project
  from_bubble: False"""
  documents = []
  for doc in raw_documents:
    doc.metadata["user"] = "Judy"
    doc.metadata["project"] = "test_project"
    doc.metadata["from_bubble"] = False
    documents.append(doc)
    
  with open("doc/"+"extractedDoc.txt","w") as file:
      for doc in documents:
        file.write(doc.doc_id)
        file.write("\n")
        file.write(doc.get_text())
        file.write('\n')
        file.write(doc.get_metadata_str())
        file.write('\n')
  print(f"Finish loading files")
  
  ### Add text nodes to vector store
  # try:
  #   cached_hashes = IngestionCache.from_persist_path(TEST_PIPELINE_CACHE)
  # except:
  #   cached_hashes = None
  # create the pipeline with transformations
  pipeline = IngestionPipeline(
    transformations=[
      SentenceSplitter(chunk_size=1024, chunk_overlap=20),
      # SummaryExtractor(llm=llm,summaries=['self']),
      OpenAIEmbedding(),
    ],
    vector_store=vector_store,
  )

  # Ingest directly into a vector db  
  nodes = pipeline.run(documents=documents)
  # pipeline.persist(TEST_PIPELINE_CACHE)
  print("Finished adding node embeddings to vectorstore")

  ## Load index from vector store
  # index = MultiModalVectorStoreIndex.from_vector_store(vector_store=vector_store, image_vector_store=image_store)
  index = MultiModalVectorStoreIndex.from_vector_store(vector_store=vector_store)
  ## Query
  openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o",
    temperature=0.0
  )

  qa_tmpl_str = (
    "Context information is below.\n"
    "-----------------------\n"
    "{context_str}\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
  )

  qa_tmpl = PromptTemplate(qa_tmpl_str)
  query_engine = index.as_query_engine(
    llm=openai_mm_llm,
    text_qa_template = qa_tmpl
    )

  test_queries = ["什么是项目名称", "气液分离器的设计目标","丝网分离器的丝直径"]
  for query_str in test_queries:
    response = query_engine.query(query_str)
    print(str(response))

  # for query_str in QUERIES:
  #   response = query_engine.query(query_str)
  #   print(str(response))
  
  
if __name__ == "__main__":
  add_text_to_index("data/combined_test/3-equipment-design.pdf")