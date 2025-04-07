import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Cargar Dataset
df = pd.read_excel(open('Dataset.xlsx', 'rb'),
                   sheet_name='Dataset')

# Ordenamos los datos en funcion del tiempo
df = df.sort_values(by=['Año', 'Cuatrimestre']).reset_index(drop=True)

# Normalizamos el tiempo en el intervalo [0,1]
df['Tiempo'] = np.linspace(1/len(df), 1, len(df))

# Variables de respuesta (Y)
Y = df[['GDP', 'Investment', 'Exports']]

# Matriz de correlacion
correlation_matrix = Y.corr()
print("\n\nMatriz Correlacion \n", correlation_matrix, "\n\n\n")

# Matriz de dispersion
sns.pairplot(df[['GDP', 'Investment', 'Exports']], diag_kind='kde')
plt.savefig("matriz_dispersion.png")

# Variables a graficar
variables = ['Exports', 'Investment', 'GDP']
titulos = ['Total export', 'Total investment', 'GDP']

# Crear subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7), sharex=True)

for i, var in enumerate(variables):
    axes[i].plot(df['Tiempo'], df[var], marker='o', linestyle='-', color='black', markersize=5, markerfacecolor='white')
    axes[i].set_ylabel(titulos[i])
    axes[i].grid(False)

# Configurar eje X
axes[-1].set_xlabel("Period")

# guardar  series
plt.tight_layout()
plt.savefig("grafico_series_temporales.png", dpi=300)

# Ajustar modelos cuadratico
X = df[['Tiempo']]

# Modelo
model = LinearRegression().fit(X, Y)

# Extraer los residuos estándar del modelo
residuals = Y.values - model.predict(X)

# Matriz de diseño X sin la constante
X_values = X.values

# Inicializar lista de residuos recursivos
recursive_residuals = []

# 2 Calculo de residuos recursivos
for j in range(1, len(X_values)):
    X_j = X_values[:j]  # Subconjunto de datos hasta j-1
    Y_j = Y.values[:j]

    # Ajustar modelo con los datos hasta la observacion j
    model_j = sm.OLS(Y_j, X_j).fit()

    # Prediccion para la observación actual
    Y_pred_j = model_j.predict(X_values[j])

    # Calcular el residuo recursivo
    e_jj = np.sqrt(1 + X_values[j] @ np.linalg.pinv(X_j.T @ X_j) @ X_values[j].T)
    u_nj = (Y.values[j] - Y_pred_j) / e_jj

    recursive_residuals.append(u_nj)

# Convertir a array de NumPy
recursive_residuals = np.array(recursive_residuals)

print("Residuos\n\n", recursive_residuals, "\n\n\n")

# 3 Construcción del proceso de suma parcial de residuos
def partial_sum_process(residuals):
    n, m = residuals.shape  # Obtener dimensiones (n: número de observaciones, m: número de variables)
    Q_n = np.zeros((n, m))  # Inicializar matriz de suma parcial

    for j in range(1, n):
        Q_n[j] = Q_n[j-1] + residuals[j]  # Sumar por cada columna

    return Q_n

# Pasar recursive_residuals de tridimensional a  bidimensional
recursive_residuals = recursive_residuals.squeeze()

# Aplicar la función a los residuos recursivos
Q_n = partial_sum_process(recursive_residuals)
print(Q_n)





