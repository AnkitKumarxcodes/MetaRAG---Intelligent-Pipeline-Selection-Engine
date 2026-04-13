# pipelines/multiquery_rag.py

from .base import BaseRAGPipeline


class MultiQueryRAG(BaseRAGPipeline):

    def generate_queries(self, query: str):
        prompt = f"""
Generate 3 different search queries for the given question.
Each query should capture a different perspective.

Return ONLY the queries, one per line.

Question:
{query}
"""

        result = self.llm.invoke(prompt).content  # ✅ FIX

        queries = [q.strip() for q in result.split("\n") if q.strip()]
        return queries

    def run(self, query: str):
        queries = self.generate_queries(query)

        all_docs = []

        for q in queries:
            docs = self.retriever.invoke(q)  # ✅ FIX (use q)
            all_docs.extend(docs)

        # remove duplicates
        unique_docs = list({d.page_content: d for d in all_docs}.values())

        context = self.format_context(unique_docs[:5])

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt).content  # ✅ FIX

        return {
            "answer": response,
            "context": context,
            "queries": queries,
            "pipeline": "multi_query"
        }