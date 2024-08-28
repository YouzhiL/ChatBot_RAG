from utils import get_doc_tools
from pathlib import Path
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.core.objects import ObjectIndex
from llama_index.core import VectorStoreIndex

paper_to_dict = {}
for paper in papers:
  print(f"Getting tools for paper: {paper}")
  vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
  paper_to_dict[paper] = [vector_tool, summary_tool]
  
all_tools = [t for paper in papers for t in paper_to_dict[paper]]

# llm = OpenAI(model="gpt-4-o")
# agent_worker = FunctionCallingAgentWorker.from_tools(
#   initial_tools,
#   llm=llm,
#   verbose=True
# )
# agent = AgentRunner(agent_worker)
# response

obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
tools = obj_retriever.retrieve("Tell me about the eval dataset used in ...")

print(tools[0].metadata)

agent_worker = FunctionCallingAgentWorker.from_tools(
  tool_retriever=obj_retriever,
  llm = llm,
  system_prompt="""\
  You are an agent designed to answer a set of given papaers. Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
  """
)


agent = AgentRunner(agent_worker)
response = agent.query(
  "Tell me about the evaluaion dataset used in MetaGPT and compare it aganst SWE-BENCH"
)


