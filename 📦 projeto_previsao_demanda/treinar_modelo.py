import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("1. Criando base de dados simulada...")
# Simulando dados de 3 anos de vendas diárias
np.random.seed(42)
datas = pd.date_range(start="2023-01-01", end="2026-10-01", freq="D")
vendas_base = np.random.randint(100, 500, size=len(datas))

df = pd.DataFrame({'data': datas, 'vendas': vendas_base})

# Adicionando efeitos de sazonalidade (Fim de semana e fim de ano vendem mais)
df['dia_da_semana'] = df['data'].dt.dayofweek
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year

df.loc[df['dia_da_semana'].isin([4, 5]), 'vendas'] += 150 # Sexta e Sábado
df.loc[df['mes'] == 12, 'vendas'] += 200 # Dezembro

print("2. Executando Engenharia de Recursos (Feature Engineering)...")
# Criando variáveis de atraso (Lag Features) para comportamento histórico
df['vendas_ontem'] = df['vendas'].shift(1)
df['vendas_7_dias_atras'] = df['vendas'].shift(7)
df['media_movel_7_dias'] = df['vendas'].shift(1).rolling(window=7).mean()

# Removendo linhas com valores nulos gerados pelo shift
df.dropna(inplace=True)

# Separando variáveis preditivas (X) e alvo (y)
X = df[['dia_da_semana', 'mes', 'ano', 'vendas_ontem', 'vendas_7_dias_atras', 'media_movel_7_dias']]
y = df['vendas']

print("3. Dividindo a base e validando com K-Fold...")
# Divisão treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurando Validação Cruzada (K-Fold com 5 quebras)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Instanciando os dois modelos mencionados na sua vaga
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

# Avaliando modelos com validação cruzada
scores_rf = cross_val_score(model_rf, X_train, y_train, cv=kf, scoring='r2')
scores_xgb = cross_val_score(model_xgb, X_train, y_train, cv=kf, scoring='r2')

print(f"-> R² Médio - Random Forest: {scores_rf.mean():.4f}")
print(f"-> R² Médio - XGBoost: {scores_xgb.mean():.4f}")

# Escolhendo o melhor modelo baseado no score e treinando no dataset completo de treino
if scores_xgb.mean() > scores_rf.mean():
    best_model = model_xgb
    print("XGBoost selecionado como melhor modelo.")
else:
    best_model = model_rf
    print("Random Forest selecionado como melhor modelo.")

best_model.fit(X_train, y_train)

# Avaliação final no conjunto de teste
predicoes = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predicoes)
r2 = r2_score(y_test, predicoes)

print(f"\n--- Métrica de Validação Final ---")
print(f"Erro Médio Absoluto (MAE): {mae:.2f} unidades")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Salvando o modelo treinado em um arquivo binário
with open('melhor_modelo_vendas.pkl', 'wb') as arquivo:
    pickle.dump(best_model, arquivo)
print("\nModelo exportado com sucesso para 'melhor_modelo_vendas.pkl'!")
