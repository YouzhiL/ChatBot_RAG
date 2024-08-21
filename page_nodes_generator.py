from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

def get_page_nodes(docs, seperator="\n---\n"):
  nodes = []
  for doc in docs:
    doc_chunks = doc.text.split(seperator)
    for doc_chunk in doc_chunks:
      node=TextNode(text=doc_chunk,metadata=deepcopy(doc.metadata))
      nodes.append(node)
  return nodes