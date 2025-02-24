import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt


# Cargar Dataset
df = pd.read_excel(open('Dataset.xlsx', 'rb'),
                   sheet_name='Dataset')

# Ordenamos los datos en funcion del tiempo
df = df.sort_values(by=['Año', 'Cuatrimestre']).reset_index(drop=True)

# Normalizamos el tiempo en el intervalo [0,1]
df['Tiempo'] = np.linspace(1/len(df), 1, len(df))
df['Tiempo^2'] = df['Tiempo']**2
df['Tiempo^3'] = df['Tiempo']**3

# Variables de respuesta (Y)
Y = df[['GDP', 'Investment', 'Exports']]

# Ajustar modelos de distintos grados
X1 = df[['Tiempo']]  # Modelo lineal
X2 = df[['Tiempo', 'Tiempo^2']]  # Modelo cuadrático
X3 = df[['Tiempo', 'Tiempo^2', 'Tiempo^3']]  # Modelo cúbico

# Ajustar modelos
model1 = LinearRegression().fit(X1, Y)
model2 = LinearRegression().fit(X2, Y)
model3 = LinearRegression().fit(X3, Y)

# Comparar modelos
print(f"R² Lineal: {model1.score(X1, Y):.4f}")
print(f"R² Cuadrático: {model2.score(X2, Y):.4f}")
print(f"R² Cúbico: {model3.score(X3, Y):.4f}")

# Funcion para calcular R2 ajustado
def r2_ajustado(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

# Funcion para calcular validacion cruzada con R2 negativo si el modelo es malo
def r2_validacion_cruzada(model, X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, Y, cv=kf, scoring='r2')
    return np.mean(scores)

# Calcular metricas
n = len(df)
p1, p2, p3 = X1.shape[1], X2.shape[1], X3.shape[1]

r2_1, r2_2, r2_3 = model1.score(X1, Y), model2.score(X2, Y), model3.score(X3, Y)
r2_adj_1, r2_adj_2, r2_adj_3 = r2_ajustado(r2_1, n, p1), r2_ajustado(r2_2, n, p2), r2_ajustado(r2_3, n, p3)
r2_cv_1, r2_cv_2, r2_cv_3 = r2_validacion_cruzada(model1, X1, Y), r2_validacion_cruzada(model2, X2, Y), r2_validacion_cruzada(model3, X3, Y)

# Imprimir resultados
print(f"Grado 1: R² = {r2_1:.4f}, R² ajustado = {r2_adj_1:.4f}, R² validación cruzada = {r2_cv_1:.4f}")
print(f"Grado 2: R² = {r2_2:.4f}, R² ajustado = {r2_adj_2:.4f}, R² validación cruzada = {r2_cv_2:.4f}")
print(f"Grado 3: R² = {r2_3:.4f}, R² ajustado = {r2_adj_3:.4f}, R² validación cruzada = {r2_cv_3:.4f}")

# Crear predicciones con cada modelo
df_sorted = df.sort_values(by='Tiempo')
X_pred = df_sorted[['Tiempo', 'Tiempo^2', 'Tiempo^3']]

# Obtener predicciones y asignarlas
preds_linear = model1.predict(df_sorted[['Tiempo']])
preds_quadratic = model2.predict(df_sorted[['Tiempo', 'Tiempo^2']])
preds_cubic = model3.predict(X_pred)

# Crear columnas separadas
df_sorted[['Linear_GDP', 'Linear_Investment', 'Linear_Exports']] = preds_linear
df_sorted[['Quadratic_GDP', 'Quadratic_Investment', 'Quadratic_Exports']] = preds_quadratic
df_sorted[['Cubic_GDP', 'Cubic_Investment', 'Cubic_Exports']] = preds_cubic

# Graficar cada variable
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

for i, var in enumerate(['GDP', 'Investment', 'Exports']):
    ax = axes[i]
    ax.scatter(df_sorted['Tiempo'], df_sorted[var], label='Datos reales', color='black', alpha=0.6)
    ax.plot(df_sorted['Tiempo'], df_sorted[f'Linear_{var}'], label='Lineal', linestyle='dashed', color='blue')
    ax.plot(df_sorted['Tiempo'], df_sorted[f'Quadratic_{var}'], label='Cuadratico', linestyle='dashed', color='green')
    ax.plot(df_sorted['Tiempo'], df_sorted[f'Cubic_{var}'], label='Cubico', linestyle='solid', color='red')

    ax.set_title(f'{var} vs. Tiempo')
    ax.set_xlabel('Tiempo (normalizado)')
    ax.set_ylabel(var)
    ax.legend()

plt.tight_layout()
plt.savefig("resultados_modelos.png")






