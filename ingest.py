import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from llm_config import get_embeddings, descrever_imagem, transcrever_audio, processar_video
import supabase_client

load_dotenv()

TRANSCRICOES_DIR = Path("./transcricoes")
CHROMA_DIR = Path("./chroma_db")


def buscar_conteudo_url(url: str) -> tuple[str, str]:
    """Faz o download de uma página e retorna (título, texto limpo)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    resposta = requests.get(url, headers=headers, timeout=30)
    resposta.raise_for_status()

    soup = BeautifulSoup(resposta.content, "html.parser")

    titulo = soup.title.get_text(strip=True) if soup.title else "Sem título"
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    texto = soup.get_text(separator="\n", strip=True)
    linhas = [linha for linha in texto.splitlines() if linha.strip()]
    texto_limpo = "\n\n".join(linhas)
    return titulo, texto_limpo


def adicionar_fonte_web(url: str, nome_arquivo: str | None = None) -> str:
    """Baixa o conteúdo de uma URL, salva em transcricoes/ e retorna o caminho salvo."""
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    titulo, conteudo = buscar_conteudo_url(url)

    if not nome_arquivo:
        nome_arquivo = urlparse(url).netloc or "fonte_web"
    nome_arquivo = re.sub(r"[^\w\-_.]", "_", nome_arquivo)
    if not nome_arquivo.endswith(".txt"):
        nome_arquivo += ".txt"

    caminho = TRANSCRICOES_DIR / nome_arquivo
    prefixo = f"Fonte: {url}\nTítulo: {titulo}\n\n"
    caminho.write_text(prefixo + conteudo, encoding="utf-8")

    supabase_client.salvar_transcricao(nome_arquivo, prefixo + conteudo, url)

    return str(caminho)


def extrair_texto_pdf(conteudo_bytes: bytes) -> str:
    """Extrai texto de um PDF a partir dos bytes do arquivo."""
    from io import BytesIO
    from pypdf import PdfReader

    leitor = PdfReader(BytesIO(conteudo_bytes))
    paginas = []
    for pagina in leitor.pages:
        texto_pagina = pagina.extract_text() or ""
        if texto_pagina.strip():
            paginas.append(texto_pagina)
    return "\n\n".join(paginas)


def adicionar_fonte_pdf(nome_arquivo: str, conteudo_bytes: bytes) -> str:
    """Extrai texto de um PDF enviado, salva em transcricoes/ e retorna o caminho salvo."""
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    texto = extrair_texto_pdf(conteudo_bytes)
    if not texto.strip():
        raise ValueError("Não foi possível extrair texto do PDF (pode ser um PDF escaneado sem OCR).")

    nome_base = re.sub(r"[^\w\-_.]", "_", Path(nome_arquivo).stem)
    nome_final = f"{nome_base}.txt"

    caminho = TRANSCRICOES_DIR / nome_final
    prefixo = f"Fonte: PDF ({nome_arquivo})\n\n"
    caminho.write_text(prefixo + texto, encoding="utf-8")

    supabase_client.salvar_transcricao(nome_final, prefixo + texto, None)

    return str(caminho)


def adicionar_fonte_imagem(nome_arquivo: str, conteudo_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Descreve/transcreve uma imagem via Gemini Vision, salva a descrição em
    transcricoes/ e a imagem original no Supabase Storage (se configurado)."""
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    descricao = descrever_imagem(conteudo_bytes, mime_type)

    nome_base = re.sub(r"[^\w\-_.]", "_", Path(nome_arquivo).stem)
    nome_final = f"{nome_base}.txt"

    caminho = TRANSCRICOES_DIR / nome_final
    prefixo = f"Fonte: Imagem ({nome_arquivo})\n\n"
    caminho.write_text(prefixo + descricao, encoding="utf-8")

    url_imagem = supabase_client.salvar_imagem(nome_arquivo, conteudo_bytes, mime_type)
    supabase_client.salvar_transcricao(nome_final, prefixo + descricao, url_imagem)

    return str(caminho)


def adicionar_fonte_audio(nome_arquivo: str, conteudo_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcreve um áudio via Gemini, salva a transcrição em transcricoes/ e
    o áudio original no Supabase Storage (se configurado)."""
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    transcricao = transcrever_audio(conteudo_bytes, mime_type)

    nome_base = re.sub(r"[^\w\-_.]", "_", Path(nome_arquivo).stem)
    nome_final = f"{nome_base}.txt"

    caminho = TRANSCRICOES_DIR / nome_final
    prefixo = f"Fonte: Áudio ({nome_arquivo})\n\n"
    caminho.write_text(prefixo + transcricao, encoding="utf-8")

    url_audio = supabase_client.salvar_audio(nome_arquivo, conteudo_bytes, mime_type)
    supabase_client.salvar_transcricao(nome_final, prefixo + transcricao, url_audio)

    return str(caminho)


def adicionar_fonte_video(nome_arquivo: str, conteudo_bytes: bytes, mime_type: str = "video/mp4") -> str:
    """Processa um vídeo via Gemini, salva a transcrição/descritivo em transcricoes/ e
    o vídeo original no Supabase Storage (se configurado)."""
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    descricao = processar_video(conteudo_bytes, mime_type)

    nome_base = re.sub(r"[^\w\-_.]", "_", Path(nome_arquivo).stem)
    nome_final = f"{nome_base}.txt"

    caminho = TRANSCRICOES_DIR / nome_final
    prefixo = f"Fonte: Vídeo ({nome_arquivo})\n\n"
    caminho.write_text(prefixo + descricao, encoding="utf-8")

    url_video = supabase_client.salvar_video(nome_arquivo, conteudo_bytes, mime_type)
    supabase_client.salvar_transcricao(nome_final, prefixo + descricao, url_video)

    return str(caminho)


def adicionar_fonte_localizacao(latitude: float, longitude: float, descricao: str | None = None) -> str:
    """Registra uma localização (lat/lon) como fonte de contexto."""
    from datetime import datetime

    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_final = f"localizacao_{agora}.txt"

    link_mapa = f"https://www.google.com/maps?q={latitude},{longitude}"
    conteudo = (
        f"Fonte: Localização registrada em {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"Coordenadas: {latitude}, {longitude}\n"
        f"Mapa: {link_mapa}\n"
    )
    if descricao:
        conteudo += f"Observação: {descricao}\n"

    caminho = TRANSCRICOES_DIR / nome_final
    caminho.write_text(conteudo, encoding="utf-8")

    supabase_client.salvar_transcricao(nome_final, conteudo, link_mapa)

    return str(caminho)


def adicionar_fonte_texto(titulo: str, conteudo: str) -> str:
    """Salva um texto livre (anotação, trecho de livro, etc.) em transcricoes/."""
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    nome_base = re.sub(r"[^\w\-_.]", "_", titulo.strip()) or "anotacao"
    nome_final = f"{nome_base}.txt"

    caminho = TRANSCRICOES_DIR / nome_final
    prefixo = f"Fonte: Texto livre ({titulo})\n\n"
    caminho.write_text(prefixo + conteudo, encoding="utf-8")

    supabase_client.salvar_transcricao(nome_final, prefixo + conteudo, None)

    return str(caminho)


def anonimizar(texto: str) -> str:
    """Aplica anonimização básica de dados sensíveis no texto."""
    # CPF: 000.000.000-00 ou 00000000000
    texto = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF_OCULTO]", texto)
    # Números de cartão de crédito (4 grupos de 4 dígitos)
    texto = re.sub(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[CARTAO_OCULTO]", texto)
    # Dados bancários genéricos
    texto = re.sub(r"\b(agência|conta)\s*[:\-]?\s*\d+[\-?\d/Xx]*", r"[DADO_BANCARIO_OCULTO]", texto, flags=re.IGNORECASE)
    # E-mails
    texto = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_OCULTO]", texto)
    # Telefones brasileiros comuns
    texto = re.sub(r"\b(?:\(?\d{2}\)?[\s-]?)?\d{4,5}[\s-]?\d{4}\b", "[TELEFONE_OCULTO]", texto)
    # Senhas
    texto = re.sub(r"\b(senha|password|pin)\s*[:\-]?\s*\S+", r"[SENHA_OCULTA]", texto, flags=re.IGNORECASE)
    return texto


def listar_arquivos_txt(diretorio: Path) -> list[Path]:
    return sorted(diretorio.glob("*.txt"))


def carregar_documentos(caminhos: list[Path]) -> list[str]:
    documentos = []
    for caminho in caminhos:
        try:
            with caminho.open("r", encoding="utf-8") as f:
                texto = f.read()
            texto = anonimizar(texto)
            documentos.append(texto)
        except Exception as e:
            print(f"Erro ao ler {caminho}: {e}")
    return documentos


def processar_e_indexar():
    if not TRANSCRICOES_DIR.exists():
        TRANSCRICOES_DIR.mkdir(parents=True, exist_ok=True)

    supabase_client.sincronizar_transcricoes_locais()

    arquivos = listar_arquivos_txt(TRANSCRICOES_DIR)
    if not arquivos:
        print(f"Nenhum arquivo .txt encontrado em {TRANSCRICOES_DIR}.")
        return

    print(f"Encontrados {len(arquivos)} arquivo(s) para ingestão.")
    textos = carregar_documentos(arquivos)

    if not textos:
        print("Nenhum texto carregado. Verifique os arquivos.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", ".", ",", " ", ""],
    )

    chunks = []
    metadatas = []
    for arquivo, texto in zip(arquivos, textos):
        chunks_do_arquivo = text_splitter.split_text(texto)
        for i, chunk in enumerate(chunks_do_arquivo):
            chunks.append(chunk)
            metadatas.append({"source": str(arquivo), "chunk_index": i})

    if not chunks:
        print("Nenhum chunk gerado.")
        return

    print(f"Gerados {len(chunks)} chunks.")

    embeddings = get_embeddings()

    print("Testando conexão com API de embeddings...")
    try:
        _ = embeddings.embed_query("teste")
        print("Conexão com embeddings OK.")
    except Exception as e:
        print(f"AVISO: Teste de embedding falhou: {e}")
        print("Continuando mesmo assim... (o erro pode ocorrer na indexação)")

    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        try:
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            print(f"Indexados {min(i + batch_size, len(chunks))} de {len(chunks)} chunks...")
        except Exception as e:
            print(f"Erro ao indexar batch {i}: {e}")
            raise

    vectorstore.persist()
    print(f"Base vetorial persistida em {CHROMA_DIR}.")


if __name__ == "__main__":
    processar_e_indexar()
