from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext,load_index_from_storage, get_response_synthesizer, set_global_handler
from llama_index.core import Settings
from llama_index.core.node_parser import TokenTextSplitter,MarkdownElementNodeParser
from llama_index.core.extractors import SummaryExtractor
from llama_parse import LlamaParse
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv
import phoenix as px


from global_setting import PIPELINE_CACHE, INDEX_STORAGE, CONVERSATION_FILE
# from page_nodes_generator import get_page_nodes
# from document_loader import load_documents
# from md_node_parser import split_to_nodes
# from index_builder import build_vec_index

def run_base_parse(file_paths):
  load_dotenv()
  embed_model = OpenAIEmbedding(model="text-embedding-3-small")
  llm = OpenAI(model="gpt-4o")
  # llm = Anthropic(model="claude-3-opus-20240229")

  ### start the Phoenix server
  px.launch_app()
  set_global_handler("arize_phoenix")
  
  ### Load documents
  documents = SimpleDirectoryReader(input_files=file_paths, filename_as_id=True).load_data()
  with open("doc/"+"extractedDoc.txt","w") as file:
    for doc in documents:
      file.write(doc.get_content())
  print(f"Finish loading files")
  
  ### Build nodes
  try:
    cached_hashes = IngestionCache.from_persist_path(PIPELINE_CACHE)
  except:
    cached_hashes = None
    
  pipeline = IngestionPipeline(
    transformations=[
      TokenTextSplitter(chunk_size=1024, chunk_overlap=20),
      SummaryExtractor(llm=llm,summaries=['self']),
      OpenAIEmbedding(),
    ],
    cache=cached_hashes
  )
  
  nodes = pipeline.run(documents=documents)
  pipeline.persist(PIPELINE_CACHE)
  print("Finish building nodes")
  
  
  
  ### Build index
  
  # try:
  #   storage_context = StorageContext.from_defaults(
  #     persist_dir=INDEX_STORAGE
  #   )
  #   vector_index = load_index_from_storage(
  #     storage_context, index_id="vector"
  #   )
  #   # tree_index = load_index_from_storage(
  #   #   storage_context, index_id="tree"
  #   # )
  #   print("All indices loaded from storage.")
  # except Exception as e:
  #   print(f"indices not found: {e}")
  #   storage_context = StorageContext.from_defaults()
  #   vector_index = VectorStoreIndex(
  #     nodes, storage_context=storage_context
  #   )
  #   vector_index.set_index_id("vector")
  #   # tree_index = TreeIndex(
  #   #   nodes, storage_context=storage_context
  #   # )
  #   # tree_index.set_index_id("tree")
  #   storage_context.persist(
  #     persist_dir=INDEX_STORAGE
  #   )
  #   print("New indexes created and persisted.")
  storage_context = StorageContext.from_defaults()
  vector_index = VectorStoreIndex(
    nodes, storage_context=storage_context
  )
  vector_index.set_index_id("vector")
  storage_context.persist(
    persist_dir=INDEX_STORAGE
  )
  print("New indexes created and persisted.")
  
  
  ### Initialize chatbot
  try:
    chat_store = SimpleChatStore.from_persist_path(CONVERSATION_FILE)
  except:
    chat_store = SimpleChatStore()  
  memory = ChatMemoryBuffer(token_limit=3000, chat_store=chat_store, chat_store_key= "0")
  
  ### Build query engine 
  storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
  index = load_index_from_storage(storage_context=storage_context, index_id="vector")
  
  # high-level API
  # query_engine = index.as_query_engine(similarity_top_k=3) # use RetrieverQueryEngine under the hood
  
  # low-level API - assemble a query engine by retriever, response_synthesizer
  retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
  response_synthesizer = get_response_synthesizer(response_mode="compact", verbose=True)
  post_processor = SimilarityPostprocessor(similarity_cutoff=0.7)
  
  query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[post_processor],
  )
  
  metadata=ToolMetadata(
    name="information_retriever",
    description="Provides official information about the document. Use a detailed plain text auestion as input to the tool"
  )
  
  query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=metadata
  )
  print("Finish building query engine")
  
  ### Build chat agent
  agent = OpenAIAgent.from_tools(
    tools=[query_engine_tool],
    llm=llm,
    memory=memory,
    system_prompt= "You are an expert of information retriever. When asked a question, you are able to find the answer in the documents provided."
  )

  prompt = input("Please enter the question. Enter q to exit: \n")
  while prompt != "q" and prompt != "Q":
    response = str(agent.chat(prompt))
    print(response)
    prompt = input("Please enter the question. Enter q to exit: \n")

  ### EVALUATION PART 
  # adapted from the examples available on the official Phoenix documentation: https://docs.arize.com/phoenix/

  from phoenix.session.evaluation import (
      get_qa_with_reference, 
      get_retrieved_documents
  )
  from phoenix.trace import DocumentEvaluations, SpanEvaluations
  from phoenix.evals import (
      HallucinationEvaluator,
      QAEvaluator,
      RelevanceEvaluator,
      OpenAIModel,
      run_evals
  )


  model = OpenAIModel(model="gpt-4o")

  retrieved_documents_df = get_retrieved_documents(px.Client())
  queries_df = get_qa_with_reference(px.Client())

  hallucination_evaluator = HallucinationEvaluator(model)
  qa_correctness_evaluator = QAEvaluator(model)
  relevance_evaluator = RelevanceEvaluator(model)

  hallucination_eval_df, qa_correctness_eval_df = run_evals(
      dataframe=queries_df,
      evaluators=[hallucination_evaluator, qa_correctness_evaluator],
      provide_explanation=True,
  )
  relevance_eval_df = run_evals(
      dataframe=retrieved_documents_df,
      evaluators=[relevance_evaluator],
      provide_explanation=True,
  )[0]

  px.Client().log_evaluations(
      SpanEvaluations(
          eval_name="Hallucination", 
          dataframe=hallucination_eval_df),
      SpanEvaluations(
          eval_name="QA Correctness", 
          dataframe=qa_correctness_eval_df),
      DocumentEvaluations(
          eval_name="Relevance", 
          dataframe=relevance_eval_df),
  )
  input("Press <ENTER> to exit")

if __name__ == "__main__":
  run_base_parse("n_beats.pdf")