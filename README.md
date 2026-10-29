# 📈 Sistema Inteligente de Previsão de Demanda Corporativa

Este projeto consiste em um pipeline end-to-end de Machine Learning voltado para a previsão de demanda de vendas. O sistema foi desenvolvido para resolver problemas reais de gestão de estoque e otimização de operações em cenários corporativos, utilizando modelos avançados de regressão.

---

## 🚀 Sobre o Projeto
Este repositório contém uma solução completa de Inteligência Artificial voltada para o mercado corporativo, focando na previsão de demanda e otimização de estoque. 

O objetivo principal foi criar um pipeline robusto e transparente, aplicando conceitos de **Engenharia de Recursos (Feature Engineering)** para capturar sazonalidades, **Validação Cruzada (K-Fold)** para garantir a estabilidade estatística entre os modelos Random Forest e XGBoost, e uma interface amigável construída em **Streamlit**. O projeto foi desenhado sob os pilares de governança e minimização de dados, garantindo conformidade com as diretrizes da **LGPD**.

---

## 🛠️ Tecnologias e Ferramentas

| Componente | Tecnologia | Finalidade |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.14 | Núcleo de desenvolvimento do sistema |
| **Modelagem** | Scikit-Learn / XGBoost | Algoritmos de Machine Learning (Random Forest e Regressão) |
| **Validação** | K-Fold Cross-Validation | Validação cruzada para mitigação de overfitting |
| **Interface** | Streamlit | Dashboard interativo para o usuário final |

---

## 📐 Arquitetura e Engenharia de Recursos

Para garantir uma alta performance preditiva, o pipeline realiza as seguintes transformações nos dados históricos:
* **Métricas Temporais:** Extração automática de sazonalidade (Mês, Ano e Dia da Semana).
* **Lag Features:** Criação de variáveis de atraso histórico (`vendas_ontem` e `vendas_7_dias_atras`).
* **Média Móvel:** Cálculo de janelas móveis de 7 dias para identificar tendências de consumo.

---

## ⚙️ Como Executar o Projeto

### 1. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 2. Treinamento e Seleção do Modelo
```bash
python treinar_modelo.py
```

### 3. Inicialização do Painel Web
```bash
python -m streamlit run app.py
```

---

## 👤 Autoria e Direitos Autorais
* **Desenvolvedor:** Jeferson (`Jefer7thson`)
* **Função:** Cientista / Analista de Dados
* **Status do Projeto:** Homologado e funcional para portfólio técnico.
