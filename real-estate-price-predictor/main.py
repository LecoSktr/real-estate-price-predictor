# Importar as bibliotecas primeiro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Criar um dataframe ficticio
data = {
    'Tamanho (m²)': [50, 80, 60, 90, 120, 70, 110, 85, 95, 75],
    'Número de Quartos': [2, 3, 2, 4, 5, 3, 4, 3, 4, 2],
    'Localização': ['Centro', 'Subúrbio', 'Centro', 'Subúrbio', 'Centro', 'Centro', 'Subúrbio', 'Subúrbio', 'Centro', 'Subúrbio'],
    'Preço (R$)': [300000, 400000, 350000, 500000, 600000, 370000, 550000, 420000, 480000, 380000]
}

df = pd.DataFrame(data)

# Convertendo a coluna 'Localização' para variáveis dummy
df = pd.get_dummies(df, columns=['Localização'], drop_first=True)

# Visualizando os dados
print("Conjunto de Dados:")
print(df)

# Plotando a correlação entre variáveis
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlação entre as Variáveis')
plt.show()

# Dividindo os dados em variáveis independentes (X) e dependente (y)
X = df.drop(columns=['Preço (R$)'])
y = df['Preço (R$)']

# Separando os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
mae = mean_absolute_error(y_test, y_pred)

print(f"Erro Médio Absoluto: R$ {mae:.2f}")

# Exibindo os coeficientes do modelo
coeficientes = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
print("\nCoeficientes do Modelo:")
print(coeficientes)

# Fazendo uma previsão com novos dados
novo_imovel = pd.DataFrame({
    'Tamanho (m²)': [85],
    'Número de Quartos': [3],
    'Localização_Subúrbio': [1]  # 1 significa "Subúrbio", 0 seria "Centro"
})

previsao = model.predict(novo_imovel)
print(f"\nPreço previsto para o novo imóvel: R$ {previsao[0]:.2f}")
