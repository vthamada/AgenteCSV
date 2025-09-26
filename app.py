# app.py

import streamlit as st
import pandas as pd
from modelos_llm import make_llm, GEMINI_MODELS, OPENAI_MODELS, OPENROUTER_MODELS
from core.perception import load_catalog_from_uploads, create_data_passport
from core.orchestrator import Orchestrator

st.set_page_config(page_title="Agente CSV", page_icon="⚙️", layout="wide")
st.title("⚙️ Agente de Análise CSV")
st.write("Carregue um ou mais arquivos CSV e faça perguntas.")

@st.cache_resource
def get_orchestrator(llm_provider: str, llm_model: str, llm_key: str | None) -> Orchestrator | None:
    """
    Cria e armazena em cache a instância principal do Orquestrador.
    A cache é invalidada se as configurações do LLM mudarem.
    """
    try:
        catalog = load_catalog_from_uploads(st.session_state.uploaded_files)
        passport = create_data_passport(catalog)
        llm = make_llm(llm_provider, llm_model, api_key=llm_key)
        return Orchestrator(llm, catalog, passport)
    except Exception as e:
        st.error(f"Erro ao inicializar o orquestrador: {e}")
        return None

def clear_chat_history():
    """Limpa o histórico de mensagens do chat."""
    st.session_state.chat_history = []

# --- Interface (Sidebar) ---

with st.sidebar:
    st.header("1. Configuração da LLM")
    prov = st.selectbox("Provedor", ["gemini", "openai", "openrouter"], key="provider")

    modelos_disponiveis = {
        "gemini": GEMINI_MODELS,
        "openai": OPENAI_MODELS,
        "openrouter": OPENROUTER_MODELS
    }
    
    opcoes_modelo = modelos_disponiveis[prov] + ["Outro (digite...)"]
    modelo_sel = st.selectbox("Modelo", opcoes_modelo, key="model_select")
    
    modelo_final = modelo_sel
    if modelo_sel == "Outro (digite...)":
        modelo_final = st.text_input("Digite o nome do modelo:", key="model_custom").strip()

    api_key_ui = st.text_input("🔑 API KEY", type="password", help="Opcional. Usa variáveis de ambiente se não for preenchido.")

    st.header("2. Upload de Dados")
    up = st.file_uploader(
        "Arquivos CSV ou ZIP", 
        type=["csv", "zip"], 
        accept_multiple_files=True,
        help="Arraste e solte ou clique para selecionar arquivos. Múltiplos CSVs ou um único ZIP são aceitos."
    )

# --- Lógica Principal ---

if up:
    st.session_state.uploaded_files = [(f.name, f.getvalue()) for f in up]
    
    if modelo_final and modelo_final != "Outro (digite...)":
        orchestrator = get_orchestrator(prov, modelo_final, api_key_ui or None)

        if orchestrator:
            st.sidebar.success(f"{len(orchestrator.catalog)} tabela(s) carregada(s).")
            with st.expander("Ver Passaporte dos Dados"):
                st.json(orchestrator.data_passport)

            st.sidebar.button("Limpar Histórico do Chat", on_click=clear_chat_history, use_container_width=True)

            st.subheader("💬 Chat Analítico")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"]["text"])
                    if "figures" in msg["content"] and msg["content"]["figures"]:
                        for fig in msg["content"]["figures"]:
                            st.pyplot(fig)
                    if isinstance(msg["content"].get("table"), pd.DataFrame): 
                        st.dataframe(msg["content"]["table"], use_container_width=True)
                    if msg["content"].get("code"):
                        with st.expander("Ver código executado"):
                            st.code(msg["content"]["code"], language="python")
            
            pergunta = st.chat_input("Faça sua pergunta sobre os dados...")
            if pergunta:
                st.session_state.chat_history.append({"role": "user", "content": {"text": pergunta}})
                with st.chat_message("user"): 
                    st.markdown(pergunta)

                with st.chat_message("assistant"):
                    agent_name_placeholder = st.empty()
                    with st.spinner("Processando..."):
                        result = orchestrator.handle_query(pergunta)
                        
                        if result.get("agent_name") == "CodeGenerationAgent" and orchestrator.code_agent.last_code:
                            result["code"] = orchestrator.code_agent.last_code
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": result})
                        
                        agent_name_placeholder.markdown(f"> *Processado por: **{result.get('agent_name', 'N/A')}***")
                        st.markdown(result["text"])
                        
                        if "figures" in result and result["figures"]:
                            for fig in result["figures"]:
                                st.pyplot(fig)
                        if isinstance(result.get("table"), pd.DataFrame):
                            st.dataframe(result["table"], use_container_width=True)
                        if result.get("code"):
                            with st.expander("Ver código executado"):
                                st.code(result["code"], language="python")
    else:
        st.warning("Por favor, selecione ou digite um modelo de LLM para continuar.")
else:
    st.info("Por favor, faça o upload de arquivos na barra lateral para começar.")