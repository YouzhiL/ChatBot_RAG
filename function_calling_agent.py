from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
### define a simple calculator tool


llm = Anthropic(model="claude-3-opus-20240229")


### Initialized Anthropic Agent
agent_worker = FunctionCallingAgentWorker.from_tools([vector_tool, summary_tool],
                                                     llm=llm,
                                                     verbose=True)
agent = AgentRunner(agent_worker)

task = agent.create_task(
  "Tell me about ... in n_beats, and then ..."
)

step_output = agnet.run_step(task.task_id)
completed_steps = agent.get_completed_steps(task.task_id)
print(f"num of completed steps for task {task.task_id}: {len(completed_steps)}")

upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")

step_output = agent.run_step(task.task_id, input="What about ...")
step_output = agent.run_step(task.task_id)
print(step_output.is_last)

response = agent.finalize_response(task.task_id)
print(str(response))
