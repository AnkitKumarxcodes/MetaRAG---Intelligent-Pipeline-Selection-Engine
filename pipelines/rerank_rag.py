# pipelines/rerank_rag.py

from .base import BaseRAGPipeline


class RerankRAG(BaseRAGPipeline):
    def rerank(self, query, docs):
        scored = []

        for d in docs:
            prompt = f"""
Rate relevance (0-10):

Query: {query}
Document: {d.page_content}

Score:
"""
            score = self.llm.invoke(prompt)

            try:
                score = float(score.strip())
            except:
                score = 0

            scored.append((score, d))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [d for _, d in scored]

    def run(self, query: str):
        docs = self.retriever.invoke(query)

        ranked_docs = self.rerank(query, docs)

        context = self.format_context(ranked_docs[:3])

        prompt = f"""
Answer the question based on context.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt).content

        return {
            "answer": response,
            "context": context,
            "pipeline": "rerank"
        }