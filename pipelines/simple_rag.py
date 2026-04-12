# pipelines/simple_rag.py

def simple_rag(query, retriever, llm):
    docs = retriever.get_top_k(query, k=3)

    context = "\n".join(docs)

    prompt = f"""
    Answer the question using context only:
    {context}

    Question: {query}
    """

    answer = llm(prompt)
    return {"answer": answer, "contexts": docs}