# router/selector.py

def pick_best(results):
    best_score = -1
    best_pipeline = None

    for name, res in results.items():
        score = (
            res["faithfulness"] +
            res["answer_relevancy"]
        )

        if score > best_score:
            best_score = score
            best_pipeline = name

    return best_pipeline