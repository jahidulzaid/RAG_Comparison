from evaluate import load

def compute_metrics(predictions, references):
    f1_metric = load("f1")
    em_metric = load("exact_match")
    
    f1 = f1_metric.compute(predictions=predictions, references=references)
    em = em_metric.compute(predictions=predictions, references=references)
    
    return {"f1": f1["f1"], "em": em["exact_match"]}
