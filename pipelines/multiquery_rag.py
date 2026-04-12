# pipelines/multiquery_rag.py

def multi_query_rag(query, retriever, llm):
    queries = [
        query,
        f"Explain: {query}",
        f"Detailed: {query}"
    ]

    docs = []
    for q in queries:
        docs.extend(retriever.get_top_k(q, k=2))

    docs = list(set(docs))  # dedupe

    context = "\n".join(docs)

    answer = llm(f"{context}\nQuestion: {query}")

    return {"answer": answer, "contexts": docs}