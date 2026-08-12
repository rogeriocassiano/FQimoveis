import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

TRANSCRICOES_DIR = Path("./transcricoes")

_client = None

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def _get_client():
    global _client
    if _client is None and SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def ativo() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _tabela(nome: str):
    client = _get_client()
    if not client:
        raise RuntimeError("Supabase não configurado.")
    return client.table(nome)


def sincronizar_transcricoes_locais():
    """Baixa todas as transcrições do Supabase para o diretório local.

    Isso garante que, no Streamlit Cloud, as transcrições sejam
    recarregadas em cada sessão a partir do banco.
    """
    if not ativo():
        return

    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        resp = _tabela("transcriptions").select("name, content").execute()
        for row in resp.data:
            caminho = TRANSCRICOES_DIR / row["name"]
            caminho.write_text(row["content"], encoding="utf-8")
    except Exception as e:
        print(f"Erro ao sincronizar transcrições do Supabase: {e}")


def carregar_mensagens(member_id: str) -> list[dict]:
    if not ativo():
        return []

    try:
        resp = _tabela("messages").select("*").eq("member_id", member_id).order("created_at").execute()
        mensagens = []
        for row in resp.data:
            msg = {"role": row["role"], "content": row["content"]}
            if row.get("fontes"):
                msg["fontes"] = row["fontes"]
            mensagens.append(msg)
        return mensagens
    except Exception as e:
        print(f"Erro ao carregar mensagens do Supabase: {e}")
        return []


def salvar_mensagem(member_id: str, role: str, content: str, fontes: list[dict] | None = None):
    if not ativo():
        return

    payload = {
        "member_id": member_id,
        "role": role,
        "content": content,
        "fontes": fontes or [],
    }
    try:
        _tabela("messages").insert(payload).execute()
    except Exception as e:
        print(f"Erro ao salvar mensagem no Supabase: {e}")


def limpar_mensagens(member_id: str):
    if not ativo():
        return

    try:
        _tabela("messages").delete().eq("member_id", member_id).execute()
    except Exception as e:
        print(f"Erro ao limpar mensagens no Supabase: {e}")


def carregar_transcricoes() -> list[tuple[str, str]]:
    if not ativo():
        return []

    try:
        resp = _tabela("transcriptions").select("name, content").execute()
        return [(row["name"], row["content"]) for row in resp.data]
    except Exception as e:
        print(f"Erro ao carregar transcrições do Supabase: {e}")
        return []


IMAGENS_BUCKET = "images"
AUDIOS_BUCKET = "audios"
VIDEOS_BUCKET = "videos"


def _salvar_arquivo_storage(bucket: str, nome_arquivo: str, conteudo_bytes: bytes, mime_type: str) -> Optional[str]:
    if not ativo():
        return None

    client = _get_client()
    try:
        caminho_remoto = f"{Path(nome_arquivo).stem}_{os.urandom(4).hex()}{Path(nome_arquivo).suffix}"
        client.storage.from_(bucket).upload(
            caminho_remoto,
            conteudo_bytes,
            {"content-type": mime_type},
        )
        return client.storage.from_(bucket).get_public_url(caminho_remoto)
    except Exception as e:
        print(f"Erro ao salvar arquivo no Supabase Storage ({bucket}): {e}")
        return None


def salvar_imagem(nome_arquivo: str, conteudo_bytes: bytes, mime_type: str = "image/jpeg") -> Optional[str]:
    """Envia a imagem para o Supabase Storage e retorna a URL pública.

    Requer um bucket público chamado 'images' criado no painel do Supabase
    (Storage → New bucket → Public bucket).
    """
    return _salvar_arquivo_storage(IMAGENS_BUCKET, nome_arquivo, conteudo_bytes, mime_type)


def salvar_audio(nome_arquivo: str, conteudo_bytes: bytes, mime_type: str = "audio/wav") -> Optional[str]:
    """Envia o áudio para o Supabase Storage e retorna a URL pública.

    Requer um bucket público chamado 'audios' criado no painel do Supabase
    (Storage → New bucket → Public bucket).
    """
    return _salvar_arquivo_storage(AUDIOS_BUCKET, nome_arquivo, conteudo_bytes, mime_type)


def salvar_video(nome_arquivo: str, conteudo_bytes: bytes, mime_type: str = "video/mp4") -> Optional[str]:
    """Envia o vídeo para o Supabase Storage e retorna a URL pública.

    Requer um bucket público chamado 'videos' criado no painel do Supabase
    (Storage → New bucket → Public bucket).
    """
    return _salvar_arquivo_storage(VIDEOS_BUCKET, nome_arquivo, conteudo_bytes, mime_type)


def salvar_transcricao(name: str, content: str, source_url: str | None = None):
    if not ativo():
        return

    try:
        _tabela("transcriptions").upsert({
            "name": name,
            "content": content,
            "source_url": source_url,
        }).execute()
    except Exception as e:
        print(f"Erro ao salvar transcrição no Supabase: {e}")
