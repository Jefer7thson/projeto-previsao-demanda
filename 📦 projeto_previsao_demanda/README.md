# 📈 Sistema Inteligente de Previsão de Demanda Corporativa

Este projeto consiste em um pipeline end-to-end de Machine Learning voltado para a previsão de demanda de vendas. O sistema foi desenvolvido para resolver problemas reais de gestão de estoque e otimização de operações em cenários corporativos, utilizando modelos avançados de regressão.

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
pip install pandas scikit-learn xgboost streamlit numpy
```

### 2. Treinamento e Seleção do Modelo
Execute o script responsável por processar os dados, aplicar a validação K-Fold e exportar o melhor modelo treinado (`.pkl`):
```bash
python treinar_modelo.py
```

### 3. Inicialização do Painel Web
Inicie a interface gráfica do Streamlit localmente na sua máquina:
```bash
python -m streamlit run app.py
```

---

## 👤 Autoria e Direitos Autorais
* **Desenvolvedor:** Jeff Tech
* **Função:** Cientista / Analista de Dados (Nível Intermediário)
* **Status do Projeto:** Homologado e funcional para portfólio técnico.
