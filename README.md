# ChatBot_RAG
Implement a chatbot augmented with customized database.

## Install

```pip install -r requirements.txt```

## Run

1. Modify `.env_example` by adding your `OPENAI_API_KEY`,`ANTHROPIC_API_KEY` and `LLAMA_CLOUD_API_KEY`. Change the file name to `.env`
2. Add PDF files to ./data folder (feel free to keep/delete the existing files)
3. Run base parser using ```python cli.py base_parse data/[file_1.pdf] data/[file_n.pdf]```


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

2. Transfomer
- `TokenTextSplitter`: Split the Document text into chunks that contain whole sentences.
- `SummaryExtractor`: Generate Summaries of the text contained by the node. (claude-3-opus-20240229 is slow in this task)
- `OpenAIEmbedding`: deault model trained to produce embeddings that effectively capture semantic meanings of the text.

2. Build Index
`VectorStoreIndex`: The workhorse in most RAG applications. Used for efficient querying because it allows for similarity searches over the embedded representations of the text.
- Similarity Search: When a query is made, the query text will be embedded and compared against the stored vectors using a similarity measure identified with cosine similarity. Cosine similarity measures the cosine of the angle between two vectors. The smaller the angle between them, the more similar they are.
- Local embedding: a well-balanced default model provided by Hugging Face.

`TreeIndex`: Generated bt LLM. Useful for summarization. (not used in base parser)

3. Store Index (TO-DO)
- Local store: persist and load from disk.
- Vector Database: Picone, CheomaDB, etc.


4. Build QueryEngine
QueryEngine is an interface that process natural language queries to generate rich responses. It often relies on one or more indexes through retrievers and can also be combined with other query engines for enhanced capabilities.
- high-level API

- low-level API

`VectorIndexRetriever`: 

`ResponseSynthesizer`: Generate responses from a language model using a user query and the retrieved context.