import string
import re

def normalize_answer(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_pred, a_true):
    return int(normalize_answer(a_pred) == normalize_answer(a_true))

def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_metrics(predictions, references):
    f1_scores = []
    em_scores = []
    for pred, ref in zip(predictions, references):
        f1_scores.append(compute_f1(pred, ref))
        em_scores.append(compute_exact(pred, ref))
    return {
        "f1": sum(f1_scores) / len(f1_scores),
        "em": sum(em_scores) / len(em_scores)
    }
