from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from retriever.embed_and_index import build_retriever

def run_rag_pipeline(question: str, file_path: str):
    retriever = build_retriever(file_path)
    
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    
    return qa.run(question)
