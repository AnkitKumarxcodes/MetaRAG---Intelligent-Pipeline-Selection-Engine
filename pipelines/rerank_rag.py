# pipelines/rerank_rag.py

def rerank_rag(query, retriever, llm):
    docs = retriever.get_top_k(query, k=8)

    # simple heuristic rerank (V1 hack)
    docs = sorted(docs, key=lambda x: len(x))[:3]

    context = "\n".join(docs)

    answer = llm(f"{context}\nQuestion: {query}")

    return {"answer": answer, "contexts": docs}