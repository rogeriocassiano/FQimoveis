"""
Configuração centralizada de provedores de LLM e embeddings.

Permite alternar entre execução 100% local (Ollama) e provedores em nuvem
(OpenAI, Groq, Google Gemini, etc.), sem exigir chave de API para embeddings
(usa HuggingFace/sentence-transformers localmente).

Variáveis de ambiente:
    EMBEDDING_PROVIDER  = "ollama" (padrão) | "huggingface" | "gemini"
    LLM_PROVIDER         = "ollama" (padrão) | "openai" | "gemini"

    OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL
    HUGGINGFACE_EMBEDDING_MODEL (padrão: sentence-transformers/all-MiniLM-L6-v2)
    OPENAI_API_KEY, OPENAI_BASE_URL (opcional, para endpoints compatíveis
        como Groq), OPENAI_MODEL (padrão: gpt-4o-mini)
    GOOGLE_API_KEY, GOOGLE_MODEL (padrão: gemini-1.5-flash)
    GOOGLE_EMBEDDING_MODEL (padrão: models/text-embedding-004)

    Observação: no Streamlit Community Cloud, prefira EMBEDDING_PROVIDER=gemini
    em vez de huggingface, pois huggingface/sentence-transformers depende de
    torch/torchvision, que podem falhar no ambiente do Cloud.
"""

import os


def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        modelo = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=modelo)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=gemini, mas GOOGLE_API_KEY não foi configurada. "
                "Adicione a chave no .env ou nos Secrets do Streamlit Cloud."
            )
        modelo = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=modelo, google_api_key=api_key)

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

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini, mas GOOGLE_API_KEY não foi configurada. "
                "Adicione a chave no .env ou nos Secrets do Streamlit Cloud."
            )
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            google_api_key=api_key,
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def descrever_imagem(imagem_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Usa o Gemini Vision para descrever/transcrever o conteúdo de uma imagem.

    Requer GOOGLE_API_KEY configurada, independentemente do LLM_PROVIDER escolhido.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Para adicionar imagens é necessário configurar GOOGLE_API_KEY "
            "(usada pelo Gemini Vision para descrever a imagem)."
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    modelo = genai.GenerativeModel(os.getenv("GOOGLE_VISION_MODEL", "gemini-1.5-flash"))

    prompt = (
        "Descreva detalhadamente o conteúdo desta imagem. Se houver texto na imagem, "
        "transcreva-o literalmente. Se for um gráfico, tabela ou documento, explique "
        "as informações apresentadas."
    )
    resposta = modelo.generate_content([
        {"mime_type": mime_type, "data": imagem_bytes},
        prompt,
    ])
    return resposta.text


def transcrever_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Usa o Gemini para transcrever/resumir o conteúdo de um áudio.

    Requer GOOGLE_API_KEY configurada, independentemente do LLM_PROVIDER escolhido.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Para adicionar áudios é necessário configurar GOOGLE_API_KEY "
            "(usada pelo Gemini para transcrever o áudio)."
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    modelo = genai.GenerativeModel(os.getenv("GOOGLE_AUDIO_MODEL", "gemini-1.5-flash"))

    prompt = (
        "Transcreva literalmente a fala contida neste áudio, em português. "
        "Se houver ruído ou trechos incompreensíveis, indique com [inaudível]."
    )
    resposta = modelo.generate_content([
        {"mime_type": mime_type, "data": audio_bytes},
        prompt,
    ])
    return resposta.text


def processar_video(video_bytes: bytes, mime_type: str = "video/mp4") -> str:
    """Usa o Gemini para transcrever/descricao o conteúdo de um vídeo.

    Requer GOOGLE_API_KEY configurada, independentemente do LLM_PROVIDER escolhido.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Para adicionar vídeos é necessário configurar GOOGLE_API_KEY "
            "(usada pelo Gemini para processar o vídeo)."
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    modelo = genai.GenerativeModel(os.getenv("GOOGLE_VIDEO_MODEL", "gemini-1.5-flash"))

    prompt = (
        "Transcreva e descreva o conteúdo deste vídeo de forma detalhada. "
        "Se houver fala, transcreva-a literalmente em português. "
        "Se houver texto na tela, transcreva-o. Descreva também ações, cenas e objetos importantes."
    )
    resposta = modelo.generate_content([
        {"mime_type": mime_type, "data": video_bytes},
        prompt,
    ])
    return resposta.text


def ambiente_configurado() -> tuple[bool, str]:
    """Verifica se há configuração mínima para rodar (local ou nuvem)."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "LLM_PROVIDER=openai, mas OPENAI_API_KEY não foi encontrada no ambiente."
        return True, ""

    if llm_provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            return False, "LLM_PROVIDER=gemini, mas GOOGLE_API_KEY não foi encontrada no ambiente."
        return True, ""

    if not os.getenv("OLLAMA_BASE_URL"):
        return False, "OLLAMA_BASE_URL não encontrado no ambiente. Configure .env ou defina LLM_PROVIDER=openai|gemini."

    return True, ""
