# evaluator/ragas_eval.py

from ragas import evaluate
from datasets import Dataset

def evaluate_output(query, answer, contexts):
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(dataset)

    return result