import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.linalg import sqrtm
from statsmodels.stats.diagnostic import het_breuschpagan
from numpy.polynomial.polynomial import polyvander

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

# Para almacenar resultados
resultados = {'Variable': [], 'Grado': [], 'AIC': [], 'KS': [], 'BP_pvalue': [], 'Coef_pvalores': []}

variables = ['GDP', 'Investment', 'Exports']

for variable in variables:
    y = datos[variable].values

    for grado in range(0, 4):
        # Crear diseño polinomial (constante + t^1 + t^2 + ... t^grado)
        X_poly = polyvander(datos['tiempo'].values, grado)
        modelo = sm.OLS(y, X_poly).fit()

        # AIC
        aic = modelo.aic

        # Prueba de Breusch-Pagan
        X_poly_const = sm.add_constant(X_poly, has_constant='add')
        bp_test = het_breuschpagan(modelo.resid, X_poly_const)
        bp_pval = bp_test[1]

        # Pruebas t (p-valores de los coeficientes)
        pvals = modelo.pvalues

        # Calcular residuos recursivos
        def residuos_recursivos_general(X, y):
            residuos = []
            for j in range(grado + 2, len(X)):
                X_j = X[:j]
                y_j = y[:j]
                modelo_j = sm.OLS(y_j, X_j).fit()
                y_pred = modelo_j.predict(X[j])
                e_jj = np.sqrt(1 + X[j] @ np.linalg.pinv(X_j.T @ X_j) @ X[j].T)
                u_j = (y[j] - y_pred) / e_jj
                residuos.append(u_j)
            return np.array(residuos)

        residuos = residuos_recursivos_general(X_poly, y)
        if len(residuos) < 2:
            continue  # evitar errores por tamaño
        # KS multivariado univariado (por variable)
        Q = np.cumsum(residuos)
        sigma = np.std(residuos)
        ks_stat = np.max(np.abs(Q / sigma))

        resultados['Variable'].append(variable)
        resultados['Grado'].append(grado)
        resultados['AIC'].append(aic)
        resultados['KS'].append(ks_stat)
        resultados['BP_pvalue'].append(bp_pval)
        resultados['Coef_pvalores'].append(pvals.tolist())

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados)
pd.set_option("display.max_colwidth", None)
# Crear tablas separadas por variable
for var in variables:
    print(f"\n{'='*60}")
    print(f"Resultados para la variable: {var}")
    print(f"{'='*60}")

    tabla_var = df_resultados[df_resultados['Variable'] == var][
        ['Grado', 'AIC', 'KS', 'BP_pvalue', 'Coef_pvalores']
    ].reset_index(drop=True)

    print(tabla_var.to_string(index=False))








