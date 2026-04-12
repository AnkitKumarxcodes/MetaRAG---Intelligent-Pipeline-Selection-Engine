# main.py

from pipelines.simple_rag import simple_rag
from pipelines.multiquery_rag import multi_query_rag
from pipelines.rerank_rag import rerank_rag

from evaluator.ragas_eval import evaluate_output
from router.selector import pick_best

def run(query, retriever, llm):

    pipelines = {
        "simple": simple_rag,
        "multi": multi_query_rag,
        "rerank": rerank_rag
    }

    results = {}

    for name, pipe in pipelines.items():
        output = pipe(query, retriever, llm)

        eval_score = evaluate_output(
            query,
            output["answer"],
            output["contexts"]
        )

        results[name] = eval_score

    best = pick_best(results)

    return best, results