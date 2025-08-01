from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI

from langchain.docstore.document import Document
import json

def load_hotpot_contexts(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    docs = []
    for entry in data:
        for title, passage_list in entry["context"]:
            # passage_list is a list of strings, join them
            passage = " ".join(passage_list)
            docs.append(Document(page_content=passage, metadata={"title": title}))
    return docs


def build_retriever(file_path: str):
    docs = load_hotpot_contexts(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding)

    return vectorstore.as_retriever()