# embeddings.py

def get_embedding(name: str):
    name = name.lower()

    if name == "nomic":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")

    elif name == "bge":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    elif name == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    else:
        raise ValueError(f"Unsupported embedding: {name}")