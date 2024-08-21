# ChatBot_RAG
Implement a chatbot augmented with customized database.

## Install

```pip install -r requirements.txt```

## Run

1. Modify `.env_example` by adding your `OPENAI_API_KEY` and `LLAMA_CLOUD_API_KEY`. Change the file name to `.env`
2. Add PDF files to ./data folder (feel free to keep/delete the existing files)
3. Run base parser using ```python cli.py base_parse data/[file_1.pdf] data/[file_n.pdf]```
