import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(page_title="Predição de Demanda", page_icon="📈", layout="centered")

# ==============================================================================
# BARRA LATERAL - COMPROVAÇÃO DE AUTORIA E GOVERNANÇA (LGPD)
# ==============================================================================
st.sidebar.image("https://flaticon.com", width=70)
st.sidebar.title("Informações do Sistema")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Desenvolvido por:**\n Jeff Tech")
st.sidebar.markdown("**Perfil Profissional:**\nCientista de Dados Intermediário")
st.sidebar.markdown("---")
st.sidebar.markdown("**Conformidade LGPD:**\nO sistema utiliza dados estritamente anonimizados e restritos ao propósito de negócio, sem armazenamento de PII (Informações Pessoais Identificáveis).")
st.sidebar.markdown("**Versão:** 1.0.0 (Homologado)")

# ==============================================================================
# CORPO PRINCIPAL DO PAINEL
# ==============================================================================
st.title("📈 Painel Inteligente de Previsão de Demanda")
st.write("Insira as métricas operacionais abaixo para calcular a previsão de vendas futuras.")

@st.cache_resource
def carregar_modelo():
    with open('melhor_modelo_vendas.pkl', 'rb') as arquivo:
        return pickle.load(arquivo)

try:
    modelo = carregar_modelo()
except FileNotFoundError:
    st.error("Erro: O arquivo 'melhor_modelo_vendas.pkl' não foi encontrado. Execute o script 'treinar_modelo.py' primeiro para gerar o modelo.")
    st.stop()

st.subheader("Parâmetros do Histórico de Vendas")

col1, col2 = st.columns(2)

with col1:
    vendas_ontem = st.number_input("Volume de vendas de ontem:", min_value=0, value=250)
    vendas_7_dias = st.number_input("Volume de vendas de 7 dias atrás:", min_value=0, value=230)
    media_7_dias = st.number_input("Média móvel de vendas (últimos 7 dias):", min_value=0.0, value=242.0)

with col2:
    dia_da_semana = st.selectbox("Dia da Semana:", 
                                 options=list(range(7)), 
                                 format_func=lambda x: ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"][x])
    mes = st.slider("Mês da Previsão:", min_value=1, max_value=12, value=10)
    ano = st.number_input("Ano da Previsão:", min_value=2026, max_value=2030, value=2026)

if st.button("Calcular Previsão de Demanda", type="primary"):
    dados_entrada = pd.DataFrame([{
        'dia_da_semana': dia_da_semana,
        'mes': mes,
        'ano': ano,
        'vendas_ontem': vendas_ontem,
        'vendas_7_dias_atras': vendas_7_dias,
        'media_movel_7_dias': media_7_dias
    }])
    
    previsao = modelo.predict(dados_entrada)
    
    st.success("Cálculo realizado com sucesso!")
    st.metric(label="Volume Estimado de Vendas para o Alvo", value=f"{int(np.round(previsao[0]))} unidades")

    
    st.info(f"Nota de Explicabilidade (XAI): Este cálculo utilizou a modelagem preditiva baseada em padrões históricos e variáveis temporais, permitindo total auditoria dos parâmetros de entrada conforme diretrizes éticas de IA.")
