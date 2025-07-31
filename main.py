from rag.pipeline import run_rag_pipeline

question = "What is the capital of France?"
file_path = "data/hotpotqa_sample.json"

answer = run_rag_pipeline(question, file_path)
print("Answer:", answer)
