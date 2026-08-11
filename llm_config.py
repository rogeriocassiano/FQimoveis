"""
Configuração centralizada de provedores de LLM e embeddings.

Permite alternar entre execução 100% local (Ollama) e provedores em nuvem
compatíveis com a API da OpenAI (OpenAI, Groq, etc.), sem exigir chave de API
para embeddings (usa HuggingFace/sentence-transformers localmente).

Variáveis de ambiente:
    EMBEDDING_PROVIDER  = "ollama" (padrão) | "huggingface"
    LLM_PROVIDER         = "ollama" (padrão) | "openai"

    OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL
    HUGGINGFACE_EMBEDDING_MODEL (padrão: sentence-transformers/all-MiniLM-L6-v2)
    OPENAI_API_KEY, OPENAI_BASE_URL (opcional, para endpoints compatíveis
        como Groq), OPENAI_MODEL (padrão: gpt-4o-mini)
"""

import os


def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        modelo = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=modelo)

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def get_chat_llm(temperature: float = 0.2):
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai, mas OPENAI_API_KEY não foi configurada. "
                "Adicione a chave no .env ou nos Secrets do Streamlit Cloud."
            )
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def ambiente_configurado() -> tuple[bool, str]:
    """Verifica se há configuração mínima para rodar (local ou nuvem)."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "LLM_PROVIDER=openai, mas OPENAI_API_KEY não foi encontrada no ambiente."
        return True, ""

    if not os.getenv("OLLAMA_BASE_URL"):
        return False, "OLLAMA_BASE_URL não encontrado no ambiente. Configure .env ou defina LLM_PROVIDER=openai."

    return True, ""
