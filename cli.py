import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from llm_config import ambiente_configurado, get_chat_llm, get_embeddings

load_dotenv()

CHROMA_DIR = Path("./chroma_db")

SYSTEM_PROMPT = """Você é o Assistente Pessoal Executivo do Gerente de Vendas da FQ Imóveis.
Seu objetivo é responder a solicitações, rascunhar mensagens e orientar tomadas de decisão com base estrita no histórico de transcrições e reuniões do gerente fornecidos no contexto.

DIRETIZES:
- Mantenha tom de voz assertivo, corporativo e focado em alta performance comercial.
- Copie os argumentos de negociação, contornos de objeção e estilo de gestão usados pelo gerente na base recuperada.
- Respeite o sigilo comercial e a LGPD: nunca exiba CPFs, dados bancários ou senhas.
- Se a informação não constar na base, informe que não há registro prévio e forneça uma sugestão alinhada ao padrão FQ Imóveis.

CONTEXTO DA BASE DO GERENTE:
{context}
"""

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + "\n\nPergunta do usuário:\n{question}\n\nResposta:",
)


def carregar_base_vetorial():
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"Base vetorial não encontrada em {CHROMA_DIR}. Execute 'reindexar' primeiro.")
        return None

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    return vectorstore


def responder(pergunta: str, vectorstore) -> str:
    resposta, _ = responder_com_fontes(pergunta, vectorstore)
    return resposta


def responder_com_fontes(pergunta: str, vectorstore) -> tuple[str, list[dict]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    fontes = []
    for doc in docs:
        source = doc.metadata.get("source", "Desconhecido")
        chunk = doc.metadata.get("chunk_index", "?")
        fontes.append({"source": source, "chunk": chunk, "content": doc.page_content[:300]})

    llm = get_chat_llm()
    chain = PROMPT_TEMPLATE | llm
    resposta = chain.invoke({"context": contexto, "question": pergunta})
    return resposta.content, fontes


def main():
    ok, mensagem = ambiente_configurado()
    if not ok:
        print(f"Erro: {mensagem}")
        print("Copie .env.example para .env e ajuste as variáveis.")
        sys.exit(1)

    print("=" * 60)
    print("Assistente Pessoal Executivo - FQ Imóveis")
    print("=" * 60)
    print("Comandos disponíveis:")
    print("  'sair'      - encerrar o assistente")
    print("  'reindexar' - reprocessar as transcrições")
    print("=" * 60)

    vectorstore = carregar_base_vetorial()

    while True:
        pergunta = input("\nVocê: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("Encerrando. Até logo!")
            break

        if pergunta.lower() == "reindexar":
            from ingest import processar_e_indexar
            processar_e_indexar()
            vectorstore = carregar_base_vetorial()
            continue

        if vectorstore is None:
            print("Base vetorial ainda não disponível. Use 'reindexar' para criá-la.")
            continue

        try:
            resposta = responder(pergunta, vectorstore)
            print(f"\nAssistente: {resposta}")
        except Exception as e:
            print(f"\nErro ao gerar resposta: {e}")


if __name__ == "__main__":
    main()
