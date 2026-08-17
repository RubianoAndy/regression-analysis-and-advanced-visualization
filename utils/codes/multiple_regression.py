"""Actividad 4 - Fase 2: regresión lineal múltiple y comparación de modelos.

Toma el diagnóstico de la Fase 1 —los residuos del modelo simple se ordenan
por sector— y lo convierte en especificación. Se estiman y comparan cuatro
modelos anidados por complejidad creciente:

    M1  costo ~ consumo                       (simple, referencia)
    M2  costo ~ consumo + sector              (rectas paralelas, ANCOVA)
    M3  costo ~ consumo * sector              (una pendiente por sector)
    M4  log(costo) ~ log(consumo) + sector    (elasticidad, escala log-log)

La comparación no se hace solo por R2: se añaden los criterios de información
de Akaike y Schwarz, el error de predicción en unidades originales y las
pruebas de supuestos, porque un R2 siempre crece al agregar regresores.

Rutas: el script se ubica en codes -> utils -> raíz del proyecto.
Lee el CSV de ``data/dataset``, escribe las tablas en ``data/processed`` y
las imágenes en ``public/assets/images/figures/python/regression/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = (
    PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python" / "regression"
)
for d in (PROCESSED_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

SECTOR_ORDER = ["Residencial", "Comercial", "Industrial"]
SECTOR_COLORS = {"Residencial": "#a6bddb", "Comercial": "#74a9cf",
                 "Industrial": "#2b8cbe"}
ACCENT = "#d95f02"
MODEL_COLORS = ["#bdd7e7", "#6baed6", "#2b8cbe", "#08519c"]

df = pd.read_csv(DATA_DIR / "consumo_energia.csv")
df["sector"] = pd.Categorical(df["sector"], categories=SECTOR_ORDER, ordered=True)
df["tarifa_cop_kwh"] = df["costo_miles_cop"] * 1000 / df["consumo_kwh"]
df["log_consumo"] = np.log(df["consumo_kwh"])
df["log_costo"] = np.log(df["costo_miles_cop"])
n = len(df)
y = df["costo_miles_cop"]

"""1. ESTIMACIÓN DE LOS CUATRO MODELOS.

``Residencial`` actúa como categoría de referencia, así que los coeficientes
de sector se leen como diferencias respecto de ese grupo. El operador ``*``
de la fórmula expande a efectos principales más interacción, que es la forma
de permitir que cada sector tenga su propia pendiente (su propia tarifa).
"""
REF = "C(sector, Treatment(reference='Residencial'))"
m1 = smf.ols("costo_miles_cop ~ consumo_kwh", data=df).fit()
m2 = smf.ols(f"costo_miles_cop ~ consumo_kwh + {REF}", data=df).fit()
m3 = smf.ols(f"costo_miles_cop ~ consumo_kwh * {REF}", data=df).fit()
m4 = smf.ols(f"log_costo ~ log_consumo + {REF}", data=df).fit()

print("=" * 72)
print("MODELO 3 - regresión múltiple con interacción consumo x sector")
print(m3.summary())

"""El modelo log-log exige dos correcciones para poder compararse con los
otros tres, porque cambia la variable dependiente:

1. Retransformación de sus predicciones con el estimador de suavizado de
   Duan; exponenciar el valor ajustado sin corregir subestima la media de una
   variable log-normal.
2. Corrección por el jacobiano de la transformación en el AIC y el BIC. La
   log-verosimilitud en la escala original es la del modelo en logaritmos
   menos la suma de log(y), de modo que AIC_original = AIC_log + 2*Σ log(y).
   Sin este ajuste el AIC de un modelo en logaritmos parece artificialmente
   bajo y la comparación carece de sentido.
"""
factor_duan = float(np.mean(np.exp(m4.resid)))
pred_m4 = np.exp(m4.fittedvalues) * factor_duan
jacobiano = float(np.sum(df["log_costo"]))


def resumen(modelo, nombre, formula, predicciones=None, correccion=0.0):
    # Métricas comparables entre modelos, siempre en unidades originales.
    pred = modelo.fittedvalues if predicciones is None else predicciones
    resid = y - pred
    bp_p = het_breuschpagan(modelo.resid, modelo.model.exog)[1]
    jb_p = jarque_bera(modelo.resid)[1]
    return {
        "modelo": nombre,
        "especificacion": formula,
        "k_parametros": int(modelo.df_model) + 1,
        "r2": round(modelo.rsquared, 4),
        "r2_ajustado": round(modelo.rsquared_adj, 4),
        "aic": round(modelo.aic + 2 * correccion, 1),
        "bic": round(modelo.bic + 2 * correccion, 1),
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 2),
        "mae": round(float(np.mean(np.abs(resid))), 2),
        "f_estadistico": round(modelo.fvalue, 1),
        "breusch_pagan_p": f"{bp_p:.2e}",
        "jarque_bera_p": f"{jb_p:.2e}",
        "durbin_watson": round(durbin_watson(modelo.resid), 3),
    }


comparacion = pd.DataFrame([
    resumen(m1, "M1 · Simple", "costo ~ consumo"),
    resumen(m2, "M2 · Múltiple aditivo", "costo ~ consumo + sector"),
    resumen(m3, "M3 · Múltiple con interacción", "costo ~ consumo × sector"),
    resumen(m4, "M4 · Log-log", "log(costo) ~ log(consumo) + sector", pred_m4,
            correccion=jacobiano),
])
comparacion.to_csv(PROCESSED_DIR / "comparacion_modelos.csv", index=False)
print("\nComparación de los cuatro modelos")
print(comparacion.drop(columns=["especificacion"]).to_string(index=False))
print(f"\nFactor de suavizado de Duan aplicado a M4: {factor_duan:.4f}")
print(f"Corrección por jacobiano sumada al AIC y al BIC de M4: "
      f"{2 * jacobiano:,.1f}")

"""2. CONTRASTE DE MODELOS ANIDADOS.

M1, M2 y M3 forman una jerarquía: cada uno añade términos al anterior. La
prueba F de Fisher decide si la mejora de ajuste compensa los grados de
libertad gastados; un p-valor bajo indica que el término añadido aporta.
"""
tabla_anova = anova_lm(m1, m2, m3)
tabla_anova.index = ["M1 · Simple", "M2 · Aditivo", "M3 · Interacción"]
tabla_anova = tabla_anova.round(4).reset_index(names="modelo")
tabla_anova.to_csv(PROCESSED_DIR / "anova_modelos.csv", index=False)
print("\nContraste F entre modelos anidados")
print(tabla_anova.to_string(index=False))

"""3. COEFICIENTES DEL MODELO MÚLTIPLE Y MULTICOLINEALIDAD.

El factor de inflación de varianza (VIF) mide cuánto se infla la varianza de
un coeficiente por su correlación con los demás regresores. En modelos con
interacción los VIF altos son estructurales —el producto consumo x sector
está correlacionado por construcción con sus factores— y no invalidan las
predicciones, solo la lectura aislada de cada coeficiente.
"""


def tabla_coeficientes(modelo, etiquetas):
    conf = modelo.conf_int(alpha=0.05)
    exog = modelo.model.exog
    # El VIF del intercepto carece de interpretación: se omite de la tabla.
    vif = [np.nan] + [variance_inflation_factor(exog, i)
                      for i in range(1, exog.shape[1])]
    return pd.DataFrame({
        "termino": etiquetas,
        "coeficiente": modelo.params.round(4).values,
        "error_std": modelo.bse.round(4).values,
        "estadistico_t": modelo.tvalues.round(2).values,
        "p_valor": [f"{p:.2e}" for p in modelo.pvalues],
        "ic95_inferior": conf[0].round(4).values,
        "ic95_superior": conf[1].round(4).values,
        "vif": np.round(vif, 2),
    })


etiquetas_m3 = [
    "Intercepto (Residencial)",
    "Sector Comercial (Δ intercepto)",
    "Sector Industrial (Δ intercepto)",
    "consumo_kwh (pendiente Residencial)",
    "consumo × Comercial (Δ pendiente)",
    "consumo × Industrial (Δ pendiente)",
]
coef_m3 = tabla_coeficientes(m3, etiquetas_m3)
coef_m3.to_csv(PROCESSED_DIR / "regresion_multiple.csv", index=False)
print("\nCoeficientes del modelo M3 con interacción")
print(coef_m3.to_string(index=False))

etiquetas_m4 = ["Intercepto (Residencial)", "Sector Comercial",
                "Sector Industrial", "log(consumo) · elasticidad"]
coef_m4 = tabla_coeficientes(m4, etiquetas_m4)
coef_m4.to_csv(PROCESSED_DIR / "regresion_loglog.csv", index=False)
print("\nCoeficientes del modelo M4 log-log")
print(coef_m4.to_string(index=False))

"""4. VALIDACIÓN EXTERNA DE LAS PENDIENTES.

Si el modelo con interacción está bien especificado, la pendiente estimada en
cada sector debe reproducir la tarifa media observada en ese sector. Es una
comprobación independiente: la tarifa se calcula directamente del cociente
costo/consumo, sin pasar por la regresión.
"""
p = m3.params
pendientes = {
    "Residencial": p["consumo_kwh"],
    "Comercial": p["consumo_kwh"]
        + p[f"consumo_kwh:{REF}[T.Comercial]"],
    "Industrial": p["consumo_kwh"]
        + p[f"consumo_kwh:{REF}[T.Industrial]"],
}
tarifas = pd.DataFrame([
    {
        "sector": s,
        "pendiente_estimada_miles_cop_kwh": round(pendientes[s], 4),
        "tarifa_implicita_cop_kwh": round(pendientes[s] * 1000, 1),
        "tarifa_media_observada_cop_kwh": round(
            df.loc[df["sector"] == s, "tarifa_cop_kwh"].mean(), 1),
    }
    for s in SECTOR_ORDER
])
tarifas["diferencia_pct"] = (
    (tarifas["tarifa_implicita_cop_kwh"]
     / tarifas["tarifa_media_observada_cop_kwh"] - 1) * 100).round(2)
tarifas.to_csv(PROCESSED_DIR / "tarifas_estimadas.csv", index=False)
print("\nPendientes del modelo frente a la tarifa observada")
print(tarifas.to_string(index=False))

"""La elasticidad del modelo log-log se contrasta contra la unidad, no contra
cero: una elasticidad de 1 significa facturación estrictamente proporcional
al consumo, que es la hipótesis de negocio interesante."""
elasticidad = m4.params["log_consumo"]
ic_elasticidad = m4.conf_int().loc["log_consumo"]
prueba_unidad = m4.t_test("log_consumo = 1")
t_unidad = float(np.ravel(prueba_unidad.tvalue)[0])
p_unidad = float(np.ravel(prueba_unidad.pvalue)[0])
print(f"\nElasticidad costo-consumo (M4): {elasticidad:.4f} "
      f"[IC 95 %: {ic_elasticidad[0]:.4f}, {ic_elasticidad[1]:.4f}]")
print(f"H0: elasticidad = 1 -> t = {t_unidad:.3f}, p = {p_unidad:.4f} -> "
      f"{'se rechaza' if p_unidad < 0.05 else 'no se rechaza'}")

"""La multicolinealidad de M3 es un artefacto de parametrización, no una
pérdida de información, y se demuestra reparametrizando. Centrar el consumo
en la media global apenas la reduce, porque aquí el problema no es la escala
sino que los tres sectores ocupan tramos de consumo casi disjuntos. Centrar
dentro de cada sector sí desacopla el término de interacción de sus factores
y hunde el VIF máximo, dejando intacto el R2: la misma información, escrita
en una base mejor condicionada.
"""
df["consumo_centrado"] = df["consumo_kwh"] - df["consumo_kwh"].mean()
df["consumo_centrado_sector"] = (
    df["consumo_kwh"]
    - df.groupby("sector", observed=True)["consumo_kwh"].transform("mean"))
variantes = [
    ("M3 · consumo sin centrar", m3),
    ("M3 · centrado en la media global",
     smf.ols(f"costo_miles_cop ~ consumo_centrado * {REF}", data=df).fit()),
    ("M3 · centrado dentro de cada sector",
     smf.ols(f"costo_miles_cop ~ consumo_centrado_sector * {REF}",
             data=df).fit()),
]
multicolinealidad = pd.DataFrame([
    {
        "especificacion": nombre,
        "vif_maximo": round(max(variance_inflation_factor(mod.model.exog, i)
                                for i in range(1, mod.model.exog.shape[1])), 2),
        "numero_condicion": round(float(mod.condition_number), 1),
        "r2": round(mod.rsquared, 6),
        "rmse": round(float(np.sqrt(np.mean(mod.resid ** 2))), 2),
    }
    for nombre, mod in variantes
])
multicolinealidad.to_csv(PROCESSED_DIR / "multicolinealidad.csv", index=False)
print("\nEfecto de la parametrización sobre la multicolinealidad")
print(multicolinealidad.to_string(index=False))

"""5. FIGURAS DE LA REGRESIÓN MÚLTIPLE."""

"""Tres rectas, una por sector: la interacción hecha gráfico. En la escala del
costo las rectas casi se superponen —el consumo domina cualquier otro efecto—,
así que el panel derecho las traduce a la magnitud que sí separa a los
sectores: la pendiente, que es la tarifa en COP por kWh."""
fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11.0, 4.2))
for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    ax.scatter(sub["consumo_kwh"], sub["costo_miles_cop"], s=26,
               color=SECTOR_COLORS[s], edgecolor="white", linewidth=0.5,
               label=f"{s} · {pendientes[s] * 1000:,.0f} COP/kWh")
    grid_s = np.linspace(sub["consumo_kwh"].min(), sub["consumo_kwh"].max(), 50)
    pred_s = m3.get_prediction(
        pd.DataFrame({"consumo_kwh": grid_s, "sector": s})).summary_frame()
    ax.plot(grid_s, pred_s["mean"], color=SECTOR_COLORS[s], lw=2.0)
    ax.fill_between(grid_s, pred_s["mean_ci_lower"], pred_s["mean_ci_upper"],
                    color=SECTOR_COLORS[s], alpha=0.35)
grid_g = np.linspace(df["consumo_kwh"].min(), df["consumo_kwh"].max(), 50)
ax.plot(grid_g, m1.predict(pd.DataFrame({"consumo_kwh": grid_g})),
        color=ACCENT, lw=1.4, linestyle="--", label="M1 · recta única")
ax.set_title("Costo frente a consumo: las tres rectas casi coinciden")
ax.set_xlabel("Consumo (kWh/mes)")
ax.set_ylabel("Costo facturado (miles de COP)")
ax.legend(fontsize=8, loc="upper left")

for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    ax_t.scatter(sub["consumo_kwh"], sub["tarifa_cop_kwh"], s=26,
                 color=SECTOR_COLORS[s], edgecolor="white", linewidth=0.5,
                 label=s)
    ax_t.hlines(pendientes[s] * 1000, sub["consumo_kwh"].min(),
                sub["consumo_kwh"].max(), color=SECTOR_COLORS[s], lw=2.4)
    ax_t.annotate(f"{pendientes[s] * 1000:,.0f}",
                  (sub["consumo_kwh"].max(), pendientes[s] * 1000),
                  xytext=(6, 4), textcoords="offset points", fontsize=8)
ax_t.axhline(m1.params["consumo_kwh"] * 1000, color=ACCENT, lw=1.4,
             linestyle="--",
             label=f"M1 · tarifa única {m1.params['consumo_kwh'] * 1000:,.0f}")
ax_t.set_title("Pendiente estimada frente a la tarifa observada")
ax_t.set_xlabel("Consumo (kWh/mes)")
ax_t.set_ylabel("Tarifa implícita (COP/kWh)")
ax_t.legend(fontsize=8, loc="upper right")
fig.suptitle("Una tarifa por sector: el modelo con interacción frente al simple",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ajuste_por_sector.png")
plt.close(fig)

"""Comparación de los cuatro modelos en tres criterios. Los R2 ajustados
difieren en la cuarta cifra decimal, así que graficarlos directamente exigiría
truncar el eje; en su lugar se grafica su complemento —la varianza que el
modelo no explica—, que tiene cero natural y hace visible la diferencia real.
Por la misma razón el BIC se expresa como distancia al mejor modelo."""
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
etiquetas = ["M1", "M2", "M3", "M4"]
series = [
    ("Varianza no explicada, 1 − R² ajustado",
     1 - comparacion["r2_ajustado"].astype(float), "%.4f"),
    ("ΔBIC respecto del mejor modelo",
     comparacion["bic"].astype(float) - comparacion["bic"].astype(float).min(),
     "%.1f"),
    ("RMSE (miles de COP)", comparacion["rmse"].astype(float), "%.1f"),
]
for ax, (titulo, valores, fmt) in zip(axes, series):
    barras = ax.bar(etiquetas, valores, color=MODEL_COLORS)
    ax.bar_label(barras, fmt=fmt, padding=2, fontsize=8)
    ax.set_title(titulo + "\n(menor es mejor)", fontsize=10)
    ax.set_xlabel("Modelo")
    ax.set_ylim(0, max(valores.max() * 1.22, 1e-9))
fig.suptitle("Los cuatro modelos bajo tres criterios de selección, todos con "
             "cero natural", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "comparacion_modelos.png")
plt.close(fig)

"""Gráfico de coeficientes con intervalos de confianza: un término es
significativo cuando su intervalo no cruza el cero. Los términos de
interacción se separan en su propio panel porque están en otra escala."""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
bloques = [
    (ax1, coef_m3.iloc[[0, 1, 2]], "Interceptos (miles de COP)"),
    (ax2, coef_m3.iloc[[3, 4, 5]], "Pendientes (miles de COP por kWh)"),
]
for ax, bloque, titulo in bloques:
    pos = np.arange(len(bloque))[::-1]
    err = np.vstack([
        bloque["coeficiente"] - bloque["ic95_inferior"],
        bloque["ic95_superior"] - bloque["coeficiente"],
    ])
    colores = ["#2b8cbe" if lo * hi > 0 else "#bdbdbd"
               for lo, hi in zip(bloque["ic95_inferior"], bloque["ic95_superior"])]
    ax.errorbar(bloque["coeficiente"], pos, xerr=err, fmt="none",
                ecolor="#6baed6", elinewidth=2, capsize=4)
    ax.scatter(bloque["coeficiente"], pos, s=55, color=colores, zorder=3)
    ax.axvline(0, color=ACCENT, lw=1.2, linestyle="--")
    ax.set_yticks(pos, [t.replace(" (", "\n(") for t in bloque["termino"]],
                  fontsize=8)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Coeficiente e intervalo de confianza al 95 %")
fig.suptitle("Coeficientes del modelo M3: azul si el intervalo excluye el cero",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "coeficientes_ic.png")
plt.close(fig)

"""Diagnóstico comparado: los mismos residuos frente a ajustados en M1, M3 y
M4. Es la forma más directa de mostrar qué corrige cada especificación."""
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=False)
for ax, (modelo, nombre, resid_col) in zip(axes, [
    (m1, "M1 · Simple", m1.resid),
    (m3, "M3 · Interacción", m3.resid),
    (m4, "M4 · Log-log", m4.resid),
]):
    for s in SECTOR_ORDER:
        m = (df["sector"] == s).to_numpy()
        ax.scatter(modelo.fittedvalues[m], resid_col[m], s=20,
                   color=SECTOR_COLORS[s], edgecolor="white", linewidth=0.4,
                   label=s)
    ax.axhline(0, color=ACCENT, lw=1.2)
    bp_p = het_breuschpagan(resid_col, modelo.model.exog)[1]
    ax.set_title(f"{nombre}\nBreusch-Pagan p = {bp_p:.3f}", fontsize=10)
    ax.set_xlabel("Valor ajustado" + (" (log)" if nombre.startswith("M4") else
                                      " (miles de COP)"))
    ax.set_ylabel("Residuo" + (" (log)" if nombre.startswith("M4") else
                               " (miles de COP)"))
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Qué corrige cada especificación: del sesgo por sector a la "
             "varianza constante", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "diagnostico_comparado.png")
plt.close(fig)

"""El modelo log-log en su propia escala: la nube se alinea sobre una recta de
pendiente prácticamente unitaria y las tres rectas quedan paralelas, porque
en logaritmos la diferencia de tarifa es un desplazamiento vertical."""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    ax1.scatter(sub["log_consumo"], sub["log_costo"], s=24,
                color=SECTOR_COLORS[s], edgecolor="white", linewidth=0.4,
                label=s)
    grid_s = np.linspace(sub["log_consumo"].min(), sub["log_consumo"].max(), 50)
    ax1.plot(grid_s,
             m4.predict(pd.DataFrame({"log_consumo": grid_s, "sector": s})),
             color=SECTOR_COLORS[s], lw=2.0)
ax1.set_title(f"Escala log-log: elasticidad = {elasticidad:.4f}")
ax1.set_xlabel("log del consumo (kWh/mes)")
ax1.set_ylabel("log del costo (miles de COP)")
ax1.legend(fontsize=8, loc="upper left")

ax2.scatter(pred_m4, y, s=24, color="#2c7fb8", edgecolor="white", linewidth=0.4)
limite = [0, float(y.max()) * 1.05]
ax2.plot(limite, limite, color=ACCENT, lw=1.4, linestyle="--",
         label="Predicción perfecta (y = ŷ)")
rmse_m4 = float(comparacion.loc[3, "rmse"])
ax2.set_title(f"Predicción retransformada a COP (RMSE = {rmse_m4:,.1f})")
ax2.set_xlabel("Costo predicho (miles de COP)")
ax2.set_ylabel("Costo observado (miles de COP)")
ax2.legend(fontsize=8, loc="upper left")
fig.suptitle("El modelo log-log: una elasticidad casi unitaria entre consumo "
             "y costo", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "loglog_elasticidad.png")
plt.close(fig)

print("\nOK - Fase 2: modelos múltiples, comparación y figuras generados")
