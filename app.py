import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from cli import carregar_base_vetorial, responder_com_fontes
from ingest import adicionar_fonte_web, adicionar_fonte_pdf, adicionar_fonte_texto, adicionar_fonte_imagem, processar_e_indexar
from llm_config import ambiente_configurado
import supabase_client

load_dotenv()

TEAM_FILE = Path("team_members.json")
TRANSCRICOES_DIR = Path("./transcricoes")
CHROMA_DIR = Path("./chroma_db")

DEFAULT_TEAM = [
    {"id": "gerente", "name": "Meu Assistente", "role": "Base de Conhecimento Pessoal", "color": "#FF6B6B"},
]


def carregar_equipe() -> list[dict]:
    if TEAM_FILE.exists():
        try:
            with TEAM_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_TEAM


def salvar_equipe(equipe: list[dict]):
    with TEAM_FILE.open("w", encoding="utf-8") as f:
        json.dump(equipe, f, ensure_ascii=False, indent=2)


def status_base() -> tuple[bool, str]:
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return True, "Pronta"
    return False, "Não indexada"


def listar_transcricoes() -> list[Path]:
    if not TRANSCRICOES_DIR.exists():
        return []
    return sorted(TRANSCRICOES_DIR.glob("*.txt"))


def inicializar_estado():
    if "team" not in st.session_state:
        st.session_state.team = carregar_equipe()
    if "member_messages" not in st.session_state:
        st.session_state.member_messages = {}
        if supabase_client.ativo():
            for membro in st.session_state.team:
                st.session_state.member_messages[membro["id"]] = supabase_client.carregar_mensagens(membro["id"])
    if "last_indexed" not in st.session_state:
        st.session_state.last_indexed = None
    if supabase_client.ativo():
        supabase_client.sincronizar_transcricoes_locais()


inicializar_estado()

st.set_page_config(page_title="Meu Assistente | Base de Conhecimento", page_icon="🧠", layout="wide")

_ok, _erro = ambiente_configurado()
if not _ok:
    st.error(f"{_erro} Configure o arquivo .env (veja .env.example) ou os Secrets do Streamlit Cloud.")
    st.stop()

# --- Barra lateral: equipe e controles ---
with st.sidebar:
    st.header("� Perfil")

    membro = st.session_state.team[0]
    if not membro:
        membro = DEFAULT_TEAM[0]

    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{membro['color']}22;"
        f"border-left:4px solid {membro['color']};'>"
        f"<strong>{membro['name']}</strong><br><span style='font-size:0.85em'>{membro['role']}</span></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.header("⚙️ Operações")

    base_ok, base_label = status_base()
    st.metric(label="Base vetorial", value=base_label)

    if st.button("Reindexar transcrições", use_container_width=True):
        with st.spinner("Processando transcrições..."):
            processar_e_indexar()
        st.session_state.last_indexed = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.success("Reindexação concluída.")

    if st.button("Limpar histórico", use_container_width=True):
        supabase_client.limpar_mensagens(membro["id"])
        st.session_state.member_messages[membro["id"]] = []
        st.rerun()

    conversa = st.session_state.member_messages.get(membro["id"], [])
    if conversa:
        texto_export = "\n\n".join(
            f"{'Usuário' if msg['role'] == 'user' else 'Assistente'}: {msg['content']}" for msg in conversa
        )
        st.download_button(
            label="Exportar conversa",
            data=texto_export,
            file_name=f"conversa_{membro['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    st.markdown("**Como usar**")
    st.markdown("1. Na aba **Fontes**, adicione sites, PDFs ou textos livres.")
    st.markdown("2. Clique em **Reindexar transcrições**.")
    st.markdown("3. Use a aba **Assistente** para conversar com o seu contexto.")

# --- Área principal ---
st.title("Meu Assistente com Contexto Próprio")
st.caption("Monte sua base de conhecimento com sites, PDFs, livros e anotações — e converse com ela")

tab_chat, tab_fontes, tab_painel = st.tabs(["💬 Assistente", "📄 Fontes", "📊 Visão Geral"])

# --- Aba Assistente ---
with tab_chat:
    mensagens = st.session_state.member_messages.setdefault(membro["id"], [])

    if not mensagens:
        st.info("Olá! Faça uma pergunta sobre o conteúdo que você adicionou na aba **Fontes**.")

    for mensagem in mensagens:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])
            if mensagem["role"] == "assistant" and "fontes" in mensagem and mensagem["fontes"]:
                with st.expander("Ver fontes utilizadas"):
                    for i, fonte in enumerate(mensagem["fontes"], start=1):
                        st.markdown(f"**{i}.** `{fonte['source']}` — chunk `{fonte['chunk']}`")
                        st.caption(fonte["content"])

    pergunta = st.chat_input("Digite sua pergunta...")

    if pergunta:
        mensagens.append({"role": "user", "content": pergunta})
        supabase_client.salvar_mensagem(membro["id"], "user", pergunta)
        with st.chat_message("user"):
            st.markdown(pergunta)

        vectorstore = carregar_base_vetorial()

        if vectorstore is None:
            resposta = "Base vetorial ainda não disponível. Clique em **Reindexar transcrições** no menu lateral."
            fontes = []
        else:
            with st.spinner("Consultando a base..."):
                try:
                    resposta, fontes = responder_com_fontes(pergunta, vectorstore)
                except Exception as e:
                    resposta = f"Erro ao gerar resposta: {e}"
                    fontes = []

        mensagens.append({"role": "assistant", "content": resposta, "fontes": fontes})
        supabase_client.salvar_mensagem(membro["id"], "assistant", resposta, fontes)
        with st.chat_message("assistant"):
            st.markdown(resposta)
            if fontes:
                with st.expander("Ver fontes utilizadas"):
                    for i, fonte in enumerate(fontes, start=1):
                        st.markdown(f"**{i}.** `{fonte['source']}` — chunk `{fonte['chunk']}`")
                        st.caption(fonte["content"])

# --- Aba Fontes ---
with tab_fontes:
    st.subheader("Adicionar contexto ao assistente")
    tab_web, tab_pdf, tab_imagem, tab_texto = st.tabs(["🌐 Site", "📄 PDF / Livro", "🖼️ Imagem", "📝 Texto livre"])

    with tab_web:
        with st.form("form_adicionar_url", clear_on_submit=True):
            url_input = st.text_input("URL do site", placeholder="https://exemplo.com/pagina")
            nome_input = st.text_input("Nome do arquivo (opcional)", placeholder="ex: mercado_imobiliario")
            enviar = st.form_submit_button("Buscar e adicionar")

        if enviar and url_input:
            with st.spinner("Buscando conteúdo da URL..."):
                try:
                    caminho_salvo = adicionar_fonte_web(url_input, nome_input or None)
                    st.success(f"Conteúdo salvo em `{caminho_salvo}`. Clique em **Reindexar transcrições** na barra lateral para incluí-lo na base.")
                except Exception as e:
                    st.error(f"Erro ao buscar URL: {e}")

    with tab_pdf:
        arquivo_pdf = st.file_uploader("Envie um PDF (livro, apostila, manual, etc.)", type=["pdf"])
        if arquivo_pdf is not None and st.button("Adicionar PDF", use_container_width=True):
            with st.spinner("Extraindo texto do PDF..."):
                try:
                    caminho_salvo = adicionar_fonte_pdf(arquivo_pdf.name, arquivo_pdf.getvalue())
                    st.success(f"Conteúdo salvo em `{caminho_salvo}`. Clique em **Reindexar transcrições** na barra lateral para incluí-lo na base.")
                except Exception as e:
                    st.error(f"Erro ao processar PDF: {e}")

    with tab_imagem:
        arquivo_imagem = st.file_uploader(
            "Envie uma imagem (foto de documento, print, gráfico, etc.)",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if arquivo_imagem is not None and st.button("Adicionar imagem", use_container_width=True):
            with st.spinner("Analisando imagem com Gemini Vision..."):
                try:
                    caminho_salvo = adicionar_fonte_imagem(
                        arquivo_imagem.name, arquivo_imagem.getvalue(), arquivo_imagem.type or "image/jpeg"
                    )
                    st.success(f"Descrição salva em `{caminho_salvo}`. Clique em **Reindexar transcrições** na barra lateral para incluí-la na base.")
                except Exception as e:
                    st.error(f"Erro ao processar imagem: {e}")

    with tab_texto:
        with st.form("form_adicionar_texto", clear_on_submit=True):
            titulo_texto = st.text_input("Título / identificação", placeholder="ex: anotacoes_reuniao")
            conteudo_texto = st.text_area("Conteúdo", placeholder="Cole aqui um trecho de livro, anotação ou qualquer texto...", height=200)
            enviar_texto = st.form_submit_button("Adicionar texto")

        if enviar_texto and titulo_texto and conteudo_texto:
            try:
                caminho_salvo = adicionar_fonte_texto(titulo_texto, conteudo_texto)
                st.success(f"Conteúdo salvo em `{caminho_salvo}`. Clique em **Reindexar transcrições** na barra lateral para incluí-lo na base.")
            except Exception as e:
                st.error(f"Erro ao salvar texto: {e}")

    st.divider()

    arquivos = listar_transcricoes()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Arquivos disponíveis")
        if not arquivos:
            st.warning(f"Nenhum arquivo `.txt` encontrado em `{TRANSCRICOES_DIR}`.")
        else:
            st.metric("Total de transcrições", len(arquivos))
            for caminho in arquivos:
                st.markdown(f"- `{caminho.name}`")

    with col2:
        st.subheader("Visualizar conteúdo")
        if arquivos:
            opcoes = [a.name for a in arquivos]
            arquivo_selecionado = st.selectbox("Selecione um arquivo", opcoes)
            caminho = TRANSCRICOES_DIR / arquivo_selecionado
            try:
                conteudo = caminho.read_text(encoding="utf-8")
                st.text_area("Conteúdo anonimizado", conteudo[:5000], height=400, disabled=True)
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")
        else:
            st.info("Adicione transcrições para visualizar o conteúdo.")

# --- Aba Painel ---
with tab_painel:
    st.subheader("Visão geral da base de conhecimento")

    total_mensagens = sum(len(msgs) for msgs in st.session_state.member_messages.values())
    total_transcricoes = len(listar_transcricoes())

    col1, col2, col3 = st.columns(3)
    col1.metric("Mensagens trocadas", total_mensagens)
    col2.metric("Transcrições", total_transcricoes)
    col3.metric("Base vetorial", "Pronta" if base_ok else "Pendente")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Última reindexação**")
        if st.session_state.last_indexed:
            st.success(st.session_state.last_indexed)
        else:
            st.warning("Ainda não reindexado nesta sessão.")

    with col_b:
        st.markdown("**Atividade recente**")
        recentes = []
        for mid, msgs in st.session_state.member_messages.items():
            nome = next((m["name"] for m in st.session_state.team if m["id"] == mid), mid)
            recentes.extend({"membro": nome, "tipo": msg["role"], "texto": msg["content"][:80]} for msg in msgs[-5:])
        recentes = sorted(recentes, key=lambda x: 0)[:5]
        if not recentes:
            st.caption("Nenhuma atividade registrada ainda.")
        else:
            for r in recentes:
                icone = "👤" if r["tipo"] == "user" else "🤖"
                st.markdown(f"{icone} **{r['membro']}**: {r['texto']}...")

    st.divider()
    st.subheader("Perfil ativo")
    m = st.session_state.team[0]
    st.markdown(
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
        f"background:{m['color']};margin-right:8px;'></span>"
        f"**{m['name']}** — {m['role']}",
        unsafe_allow_html=True,
    )
