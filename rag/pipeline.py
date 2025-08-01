
from langchain.chains import RetrievalQA
from retriever.embed_and_index import build_retriever
from langchain_community.llms import Ollama

def run_rag_pipeline(question: str, file_path: str):
    retriever = build_retriever(file_path)

    llm = Ollama(
        base_url="http://103.102.42.109:11434",
        model="llama3.1:8b"

    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa.run(question)
