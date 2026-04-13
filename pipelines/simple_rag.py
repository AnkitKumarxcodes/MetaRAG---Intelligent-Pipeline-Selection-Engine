# pipelines/simple_rag.py

from .base import BaseRAGPipeline


class SimpleRAG(BaseRAGPipeline):
    def run(self, query: str):
        docs = self.retriever.invoke(query)

        context = self.format_context(docs)

        prompt = f"""
Answer the question based ONLY on the context.

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
            "docs": docs,
            "pipeline": "simple"
        }