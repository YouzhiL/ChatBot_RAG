# ChatBot_RAG
Implement a chatbot augmented with customized database.

## Local Environment Install

```pip install -r requirements.txt```

## Docker Install
```docker build -t llama-index-app .```

```docker run --rm -it llama-index-app:latest /bin/bash```

## Run

1. Modify `.env_example` by adding your `OPENAI_API_KEY`,`ANTHROPIC_API_KEY` and `LLAMA_CLOUD_API_KEY`. Change the file name to `.env`
2. Add PDF files to ./data folder (feel free to keep/delete the existing files)
3. Run base parser using ```python cli.py base_parse data/[file_1.pdf] data/[file_n.pdf]```
4. Run llama parser using ```python cli.py llama_parse data/[file_1.pdf] data/[file_n.pdf]```

## Tracing & Evaluate RAG workflow
- We trace and evaluate the RAG application by using the Phoenix framework. 
- Use URL `http://localhost:6006/` to get a decomposed view of the query engine operation.
- Evaluation Standard: Retrieval quality; Generation quality; Faithfulness; Efficiency; Robustness
- Phoenix as the evaluation framework: use Chat GPT4 as a reference to decide on the correctness of RAG's answer
- [How it works](https://docs.arize.com/phoenix/evaluation/concepts-evals/llm-as-a-judge)
- Intermediate result is also stored in folder `intermediate result`.


## Discussion - BaseParse Mode
1. File Loader
- `SimpleDirectoryReader`: able to adapt to different file types and select tools to extract the content based on the file extensions. For binary files such as PDF and Offices docs, it uses PyPDF and Pillow.

2. Transformer
- `TokenTextSplitter`: Split the Document text into chunks that contain whole sentences.
- `SummaryExtractor`: Generate Summaries of the text contained by the node. (claude-3-opus-20240229 is slow in this task)
- `OpenAIEmbedding`: deault model trained to produce embeddings that effectively capture semantic meanings of the text.

3. Build Index
`VectorStoreIndex`: The workhorse in most RAG applications. Used for efficient querying because it allows for similarity searches over the embedded representations of the text.
- Similarity Search: When a query is made, the query text will be embedded and compared against the stored vectors using a similarity measure identified with cosine similarity. Cosine similarity measures the cosine of the angle between two vectors. The smaller the angle between them, the more similar they are.
- Local embedding: a well-balanced default model provided by Hugging Face.

`TreeIndex`: Generated bt LLM. Useful for summarization. (not used in base parser)

4. Store Index (TO-DO)
- Local store: persist and load from disk.
- Vector Database: Picone, CheomaDB, etc.


5. Build QueryEngine
QueryEngine is an interface that process natural language queries to generate rich responses. It often relies on one or more indexes through retrievers and can also be combined with other query engines for enhanced capabilities.
- high-level API

- low-level API
typical steps in query process: retrieval, postprocessing, and response synthesis
  - Retrievers: Browse an index and select the relevant nodes to build the necessary context. A retriever will return a `NodeWithScore` object - a structure that combines a node with an associated score. The retrival modes that can be selected is usually dependent on the index. In the case of `VectorStoreIndex`, we can choose retriever from `VectorIndexRetriever` and `VectorIndexAutoRetriever`
    `VectorIndexRetriever`: Default retriever that's used by `VectorStoreIndex`. This retriever is able to narrow down the search scope of the retriever, sets query mode of the vector store, and return the nodes with `top_k` similarity.
  - Postprocessor: Adjust the retrieved context before that context gets injected into a prompt and sent to LLM for response synthesis. They operate by either filtering, transforming, or re-ranking nodes.p
  - Synthesizer:
    `compact` mode: Similar to `refine` mode, but with reduced number of required LLM queries.

## Discussion - LlamaParse Mode

### Result
- Q1 What is the formula used in basic building block
  The formula used in the basic building block involves a series of fully connected layers with RELU non-linearities, followed by linear projection layers. The operation of the first part of the block is described by the following equations:

  1. \( h'_{1} = \text{RELU}(W'_{1}x' + b'_{1}) \)
  2. \( h'_{2} = \text{RELU}(W'_{2}h'_{1} + b'_{2}) \)
  3. \( h'_{3} = \text{RELU}(W'_{3}h'_{2} + b'_{3}) \)
  4. \( h'_{4} = \text{RELU}(W'_{4}h'_{3} + b'_{4}) \)
  5. \( b = W'b(h'_{4}) \)
  6. \( \theta_{f} = W'f(h'_{4}) \)

  These equations describe the transformations applied to the input \( x' \) through a series of layers to produce the final outputs.

- Q2 what is the performance using Statistical method on M4 test sets?                                                
  The performance using the statistical method on the M4 test sets is summarized as follows:

  - **Yearly**: 
    - OWA (Overall Weighted Average): 0.788
    - M4 Rank: 8

  - **Quarterly**: 
    - OWA: 0.898
    - M4 Rank: 8

  - **Monthly**: 
    - OWA: 0.905
    - M4 Rank: 8

  - **Others**: 
    - OWA: 0.989
    - M4 Rank: 8

  - **Average**: 
    - OWA: 0.861

  These metrics indicate the performance of the statistical method across different frequencies of the M4 test sets.

- Q3 How many Yearly/6 time series of Finance type are in the M4 dataset?
  There are 6,519 Yearly/6 time series of the Finance type in the M4 dataset.

- Q4 What are the possible topologies for N-BEATS?
  The possible topologies for N-BEATS are:

  1. N-BEATS-DRESS
  2. PARALLEL
  3. NO-RESIDUAL
  4. LAST-FORWARD
  5. NO-RESIDUAL-LAST-FORWARD