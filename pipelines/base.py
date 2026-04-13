# pipelines/base.py

from abc import ABC, abstractmethod


class BaseRAGPipeline(ABC):
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    @abstractmethod
    def run(self, query: str):
        pass

    def format_context(self, docs):
        return "\n\n".join([d.page_content for d in docs])