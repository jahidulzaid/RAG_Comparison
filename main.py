from rag.pipeline import run_rag_pipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader


question = "What is the capital of Bangladesh?"
file_path = "data/hotpot_qa.json"

answer = run_rag_pipeline(question, file_path)
print("Answer:", answer)
