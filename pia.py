import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

# Crear predicciones con cada modelo
df_sorted = df.sort_values(by='Tiempo')  # Asegurar orden correcto
X_pred = df_sorted[['Tiempo', 'Tiempo^2', 'Tiempo^3']]

# Obtener predicciones y asignarlas correctamente
preds_linear = model1.predict(df_sorted[['Tiempo']])
preds_quadratic = model2.predict(df_sorted[['Tiempo', 'Tiempo^2']])
preds_cubic = model3.predict(X_pred)

# Crear columnas separadas
df_sorted[['Linear_GDP', 'Linear_Investment', 'Linear_Exports']] = preds_linear
df_sorted[['Quadratic_GDP', 'Quadratic_Investment', 'Quadratic_Exports']] = preds_quadratic
df_sorted[['Cubic_GDP', 'Cubic_Investment', 'Cubic_Exports']] = preds_cubic

# Graficar cada variable por separado
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
plt.savefig("resultados.png")






