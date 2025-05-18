import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.linalg import sqrtm
from statsmodels.stats.diagnostic import het_breuschpagan
from numpy.polynomial.polynomial import polyvander

pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', 120)
pd.set_option('display.precision', 4)

datos = pd.read_excel(open('Dataset.xlsx', 'rb'), sheet_name='Dataset')
datos = datos.sort_values(by=['Año', 'Cuatrimestre']).reset_index(drop=True)

# Variable de tiempo normalizada
datos['tiempo_norm'] = np.linspace(1/len(datos), 1, len(datos))
variables_respuesta_nombres = ['GDP', 'Investment', 'Exports']
respuesta_df = datos[variables_respuesta_nombres]

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=respuesta_df, palette="pastel")
plt.title("Boxplot de Variables de Respuesta")
plt.ylabel("Valor")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("boxplot_atipicos.png")
plt.close()

# Matriz de correlacion
matriz_correlacion = respuesta_df.corr()
print("\nMatriz de correlacion:\n", matriz_correlacion, "\n")


# Matriz de dispersion
pair_plot = sns.pairplot(respuesta_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'})
pair_plot.fig.suptitle("Matriz de Dispersión de Variables de Respuesta", y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig("matriz_dispersion.png")
plt.close()

# Graficar las series de tiempo
fig, ejes = plt.subplots(nrows=len(variables_respuesta_nombres), ncols=1, figsize=(12, 8), sharex=True)
for i, var_nombre_plot in enumerate(variables_respuesta_nombres):
    ejes[i].plot(datos['tiempo_norm'], datos[var_nombre_plot], marker='o', linestyle='-', color='dodgerblue', markersize=4, markerfacecolor='lightblue', linewidth=1.5)
    ejes[i].set_ylabel(var_nombre_plot)
    ejes[i].grid(True, linestyle=':', alpha=0.6)
ejes[-1].set_xlabel("Periodo Normalizado (Tiempo)")
fig.suptitle("Series Temporales de Variables de Respuesta", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("series_temporales.png", dpi=300)
plt.close()

def calcular_residuos_recursivos_univariado(X_design, y_target, min_obs_para_inicio):
    residuos_rec = []
    n_obs, n_params = X_design.shape
    start_j = max(n_params, min_obs_para_inicio)

    if start_j >= n_obs:
        return np.array([])

    for j_idx_actual_obs in range(start_j, n_obs):
        X_subset_ajuste = X_design[:j_idx_actual_obs, :]
        y_subset_ajuste = y_target[:j_idx_actual_obs]
        x_actual_para_pred = X_design[j_idx_actual_obs, :].reshape(1, -1)
        y_actual_obs = y_target[j_idx_actual_obs]

        if X_subset_ajuste.shape[0] < n_params:
            continue

        modelo_subset = sm.OLS(y_subset_ajuste, X_subset_ajuste).fit(disp=0)
        y_pred_val = modelo_subset.predict(x_actual_para_pred)[0]

        pinv_XsubsetT_Xsubset = np.linalg.pinv(X_subset_ajuste.T @ X_subset_ajuste)
        leverage_term_matrix = x_actual_para_pred @ pinv_XsubsetT_Xsubset @ x_actual_para_pred.T
        e_jj_val = np.sqrt(1 + leverage_term_matrix.item())

        u_j_val = (y_actual_obs - y_pred_val) / e_jj_val
        residuos_rec.append(u_j_val)

    return np.array(residuos_rec)

# Analisis de modelos
resultados_consolidados_por_grado = []
detalles_univariados_todos_modelos = []

grados_a_probar = [0, 2, 3, 4]
tiempo_para_poly_ajuste = datos['tiempo_norm'].values

for grado_actual_comun in grados_a_probar:
    X_matriz_poly_comun = polyvander(tiempo_para_poly_ajuste, grado_actual_comun)
    n_params_actual = X_matriz_poly_comun.shape[1]

    residuos_recursivos_para_U = []
    aic_sum_para_grado_actual = 0.0
    bp_pvalues_grado_actual_lista = []

    for var_nombre_actual in variables_respuesta_nombres:
        y_serie_actual = datos[var_nombre_actual].values
        modelo_ols_univar = sm.OLS(y_serie_actual, X_matriz_poly_comun).fit()

        aic_univar = modelo_ols_univar.aic
        aic_sum_para_grado_actual += aic_univar

        bp_lm_pvalor_univar = np.nan
        if X_matriz_poly_comun.shape[1] > 1:
            bp_test_stats = het_breuschpagan(modelo_ols_univar.resid, modelo_ols_univar.model.exog)
            bp_lm_pvalor_univar = bp_test_stats[1]
        bp_pvalues_grado_actual_lista.append(bp_lm_pvalor_univar)

        coef_pvals_univar = modelo_ols_univar.pvalues

        detalles_univariados_todos_modelos.append({
            "Variable": var_nombre_actual,
            "Grado_Polinomio": grado_actual_comun,
            "AIC_Univar": round(aic_univar, 2),
            "BP_Univar": round(bp_lm_pvalor_univar, 4) if not np.isnan(bp_lm_pvalor_univar) else np.nan,
            "Coef_Pvalores_Univar": [round(p, 4) for p in coef_pvals_univar]
        })

        res_rec_univar = calcular_residuos_recursivos_univariado(X_matriz_poly_comun, y_serie_actual, min_obs_para_inicio=n_params_actual)
        if res_rec_univar.size > 0:
            residuos_recursivos_para_U.append(res_rec_univar)

    ks_multivariado_calculada = np.nan
    num_res_rec_efectivos_ks = 0

    if len(residuos_recursivos_para_U) == len(variables_respuesta_nombres):
        residuos_validos_para_U = [res_arr for res_arr in residuos_recursivos_para_U if res_arr.size > 0]
        if len(residuos_validos_para_U) == len(variables_respuesta_nombres):
            min_len_res = min(len(res_arr) for res_arr in residuos_validos_para_U)
            if min_len_res >= 2:
                num_res_rec_efectivos_ks = min_len_res
                U_matrix = np.column_stack([res_arr[:min_len_res] for res_arr in residuos_validos_para_U])

                Sigma_hat = np.cov(U_matrix.T)
                if Sigma_hat.ndim == 0:
                    Sigma_hat_inv_sqrt = 1/np.sqrt(Sigma_hat) if Sigma_hat > 1e-9 else np.array([[np.inf]])
                elif np.linalg.det(Sigma_hat) < 1e-12:
                    Sigma_hat_inv_sqrt = np.linalg.pinv(sqrtm(Sigma_hat))
                else:
                    Sigma_hat_inv_sqrt = np.linalg.inv(sqrtm(Sigma_hat))

                Q_matrix = np.cumsum(U_matrix, axis=0)

                if Sigma_hat_inv_sqrt.ndim < 2 and Sigma_hat_inv_sqrt.size == 1:
                    Q_estandar_matrix = Q_matrix * Sigma_hat_inv_sqrt.item()
                else:
                    Q_estandar_matrix = Q_matrix @ Sigma_hat_inv_sqrt

                normas_l2_por_tiempo = np.linalg.norm(Q_estandar_matrix, axis=1)
                if normas_l2_por_tiempo.size > 0:
                    ks_multivariado_calculada = np.max(normas_l2_por_tiempo)

    resultados_consolidados_por_grado.append({
        "Grado_Polinomio": grado_actual_comun,
        "AIC_Agregado": round(aic_sum_para_grado_actual, 2),
        "KS_Multivariada": round(ks_multivariado_calculada, 4) if not np.isnan(ks_multivariado_calculada) else np.nan,
        "Num_Res_Rec_KS": num_res_rec_efectivos_ks,
        "BP_Univar (GDP,Inv,Exp)": [round(p,4) if not np.isnan(p) else np.nan for p in bp_pvalues_grado_actual_lista]
    })

# --- Resultados ---
print("\n--- Resultados consolodidados por grado polinomico ---")
df_resultados_finales_grado = pd.DataFrame(resultados_consolidados_por_grado)
if not df_resultados_finales_grado.empty:
    df_resultados_finales_grado = df_resultados_finales_grado.set_index("Grado_Polinomio")
print(df_resultados_finales_grado.to_string())

print("\n--- Detalles ajuste univariado (AIC, BP, Coef. Significancia) ---")
df_detalles_univar_final = pd.DataFrame(detalles_univariados_todos_modelos)
if not df_detalles_univar_final.empty:
    for var_nombre_actual_loop in variables_respuesta_nombres:
        print(f"\n  Detalles para Variable: {var_nombre_actual_loop}")
        tabla_var_detalle = df_detalles_univar_final[df_detalles_univar_final['Variable'] == var_nombre_actual_loop][
            ['Grado_Polinomio', 'AIC_Univar', 'BP_Univar', 'Coef_Pvalores_Univar']
        ].set_index('Grado_Polinomio')

        def format_pvals(pvals_list_format):
            if not isinstance(pvals_list_format, list): return "N/A"
            return ", ".join([f"B{i}={p:.3f}{'*' if p < 0.05 else ''}" for i, p in enumerate(pvals_list_format)])

        tabla_var_detalle['Coef_Pvalores_Univar'] = tabla_var_detalle['Coef_Pvalores_Univar'].apply(format_pvals)
        print(tabla_var_detalle.to_string())
else:
    print("No se generaron resultados detallados univariados.")




