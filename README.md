# RAG_Comparison

## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, HuggingFace embeddings, FAISS vector store, and Ollama LLM. It is designed to answer questions using a HotpotQA-style dataset and provides evaluation scripts for F1 and Exact Match metrics. The code is modular and can be adapted to compare different retrievers, embeddings, or LLMs.

## Features
- Loads and indexes HotpotQA-style datasets
- Uses HuggingFace sentence-transformers for embeddings
- FAISS for fast vector search
- Ollama for local LLM inference (configurable)
- Evaluation scripts for F1 and Exact Match
- Progress bar for batch evaluation

## Environment Setup (Conda)

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd RAG_Comparison
   ```

2. **Create and activate a conda environment:**
   ```sh
   conda create -n rag_env python=3.10 -y
   conda activate rag_env
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **(Optional) Download Ollama and required models:**
   - Install Ollama from https://ollama.com/
   - Pull your desired model, e.g.:
     ```sh
     ollama pull llama3:8b
     ollama serve
     ```

## Usage

### Run a single question
```sh
python main.py
```

### Evaluate on a dataset
```sh
python evaluation/evaluate_rag.py
```

- Edit `evaluation/evaluate_rag.py` to change the number of questions or dataset path.

## Data Format
- Expects HotpotQA-style JSON files in `data/`.
- Each entry should have `question`, `answer`, and `context` fields.

## Notes
- For best performance, use a GPU and a quantized or smaller LLM model.
- The first run may be slow due to model downloads and initialization.
- For large datasets, adjust batch size or number of questions in the evaluation script.

---

**Author:** jahidulzaid
# RAG_Comparison
