import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.linalg import sqrtm

# cargar los datos
datos = pd.read_excel(open('Dataset.xlsx', 'rb'), sheet_name='Dataset')

# ordenar los datos por año y cuatrimestre
datos = datos.sort_values(by=['Año', 'Cuatrimestre']).reset_index(drop=True)

# crear la variable tiempo normalizada en el intervalo [0,1]
datos['tiempo'] = np.linspace(1/len(datos), 1, len(datos))

# seleccionar las variables de respuesta
respuesta = datos[['GDP', 'Investment', 'Exports']]

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=respuesta, palette="pastel")
plt.title("Boxplot - Detección de valores atípicos")
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_atipicos.png")

# matriz de correlacion
matriz_correlacion = respuesta.corr()
print("\nMatriz de correlacion:\n", matriz_correlacion, "\n")

# matriz de dispersion
sns.pairplot(respuesta, diag_kind='kde')
plt.savefig("matriz_dispersion.png")

# graficar las series de tiempo
variables = ['Exports', 'Investment', 'GDP']
titulos = ['Exports', 'Investment', 'GDP']

fig, ejes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7), sharex=True)
for i, var in enumerate(variables):
    ejes[i].plot(datos['tiempo'], datos[var], marker='o', linestyle='-', color='black', markersize=5, markerfacecolor='white')
    ejes[i].set_ylabel(titulos[i])
    ejes[i].grid(False)
ejes[-1].set_xlabel("Periodo")
plt.tight_layout()
plt.savefig("series_temporales.png", dpi=300)

# modelo lineal
X = datos[['tiempo']]
X_lineal = sm.add_constant(X)

# funcion para calcular residuos recursivos
def calcular_residuos_recursivos(X, y):
    residuos = []
    for j in range(2, len(X)):
        X_j = X[:j]
        y_j = y[:j]
        modelo_j = sm.OLS(y_j, X_j).fit()
        y_pred = modelo_j.predict(X[j])
        e_jj = np.sqrt(1 + X[j] @ np.linalg.pinv(X_j.T @ X_j) @ X[j].T)
        u_j = (y[j] - y_pred) / e_jj
        residuos.append(u_j)
    return np.array(residuos)

# residuos recursivos por variable
res_gdp = calcular_residuos_recursivos(X_lineal.values, datos['GDP'].values)
res_investment = calcular_residuos_recursivos(X_lineal.values, datos['Investment'].values)
res_export = calcular_residuos_recursivos(X_lineal.values, datos['Exports'].values)

# igualar tamaño minimo
longitud = min(len(res_gdp), len(res_investment), len(res_export))
U = np.column_stack((res_gdp[:longitud], res_investment[:longitud], res_export[:longitud]))

# suma parcial
def suma_parcial(residuos):
    n, p = residuos.shape
    Q = np.zeros((n, p))
    for j in range(1, n):
        Q[j] = Q[j-1] + residuos[j]
    return Q

Q = suma_parcial(U)

# estimar la matriz de covarianza
Sigma = np.cov(U.T)
Sigma_inv_sqrt = np.linalg.inv(sqrtm(Sigma))

# transformar los residuos
Q_estandar = Q @ Sigma_inv_sqrt.T

# calcular la estadistica KS
ks = np.max(np.linalg.norm(Q_estandar, axis=1))
print("\nEstadistica KS:", ks)








