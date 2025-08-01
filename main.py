from rag.pipeline import run_rag_pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader

import warnings
warnings.filterwarnings("ignore")



question = "What is the capital of Bangladesh?"
file_path = "data/hotpotqa_sample.json"

answer = run_rag_pipeline(question, file_path)
print("Answer:", answer)
