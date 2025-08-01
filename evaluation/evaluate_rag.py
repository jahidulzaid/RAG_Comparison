import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from rag.pipeline import run_rag_pipeline
from evaluation.metrics import compute_metrics



import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

# Load your dataset
with open("data/hotpotqa_sample.json") as f:
    data = json.load(f)

questions = [item["question"] for item in data[:1]]
references = [item["answer"] for item in data[:1]]

predictions = []
for q in tqdm(questions, desc="Evaluating"):
    result = run_rag_pipeline(q, "data/hotpotqa_sample.json")
    # If your pipeline returns a dict, extract the 'result' field
    if isinstance(result, dict) and 'result' in result:
        predictions.append(result['result'])
    else:
        predictions.append(result)

def ref_to_str(ref):
    if isinstance(ref, list):
        return " ".join(str(x) for x in ref)
    return str(ref)

predictions = [str(p) for p in predictions]
references = [ref_to_str(r) for r in references]
metrics = compute_metrics(predictions, references)
print("F1:", metrics["f1"])
print("Exact Match:", metrics["em"])
