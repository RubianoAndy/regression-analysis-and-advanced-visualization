"""Fase 2 - Regresión lineal simple y múltiple con statsmodels.

Responde dos preguntas encadenadas:

1. ¿Cuánto del precio explica el área por sí sola?     -> modelo simple
2. ¿Cuánto se gana al añadir las otras tres variables? -> modelo múltiple

Ambos modelos se ajustan por mínimos cuadrados ordinarios (MCO) y se comparan
con R², R² ajustado y el error medio de predicción. Las figuras se hacen con
Matplotlib.

Ejecutar desde la raíz del proyecto:
    python utils/codes/python/regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "data" / "dataset" / "viviendas.csv"
TABLAS = PROJECT_ROOT / "data" / "processed"
FIGURAS = (PROJECT_ROOT / "public" / "assets" / "images" / "figures"
           / "python" / "regression")
for carpeta in (TABLAS, FIGURAS):
    carpeta.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})
AZUL, NARANJA, GRIS = "#2b8cbe", "#d95f02", "#9e9e9e"

if not DATASET.exists():
    raise SystemExit(
        "Falta el dataset. Ejecuta antes: python utils/codes/python/dataset.py")
df = pd.read_csv(DATASET)
OBJETIVO = "precio_millones_cop"

# 1. Correlación: qué tan asociada está cada variable con el precio.
# El coeficiente de Pearson mide la fuerza y el signo de la relación lineal.
# Sirve para ordenar las variables candidatas antes de modelar nada.
PREDICTORAS = ["area_m2", "habitaciones", "antiguedad_anios", "estrato"]
filas = []
for variable in PREDICTORAS:
    r, p_valor = stats.pearsonr(df[variable], df[OBJETIVO])
    filas.append({
        "variable": variable,
        "pearson_r": round(r, 4),
        "p_valor": f"{p_valor:.2e}",
        "relacion": "positiva" if r > 0 else "negativa",
    })
correlaciones = pd.DataFrame(filas).sort_values(
    "pearson_r", key=abs, ascending=False)
correlaciones.to_csv(TABLAS / "correlaciones.csv", index=False)
print("Correlación de cada variable con el precio")
print(correlaciones.to_string(index=False))

# 2. Regresión lineal simple: precio ~ área.
# El área es la variable más correlacionada, así que es el punto de partida
# natural. La pendiente se lee directamente como millones de COP por m².
simple = smf.ols(f"{OBJETIVO} ~ area_m2", data=df).fit()
print("\n" + "=" * 70)
print(simple.summary())

ic_simple = simple.conf_int()
coef_simple = pd.DataFrame({
    "termino": ["Intercepto", "area_m2"],
    "coeficiente": simple.params.round(3).values,
    "error_estandar": simple.bse.round(3).values,
    "estadistico_t": simple.tvalues.round(2).values,
    "p_valor": [f"{p:.2e}" for p in simple.pvalues],
    "ic95_inferior": ic_simple[0].round(3).values,
    "ic95_superior": ic_simple[1].round(3).values,
})
coef_simple.to_csv(TABLAS / "regresion_simple.csv", index=False)
print("\nCoeficientes del modelo simple")
print(coef_simple.to_string(index=False))

# 3. Regresión lineal múltiple: se añaden las tres variables restantes.
# Cada coeficiente se interpreta "manteniendo constantes las demás variables",
# que es justo lo que el modelo simple no puede hacer.
multiple = smf.ols(f"{OBJETIVO} ~ " + " + ".join(PREDICTORAS), data=df).fit()
print("\n" + "=" * 70)
print(multiple.summary())

ic_multiple = multiple.conf_int()
coef_multiple = pd.DataFrame({
    "termino": ["Intercepto"] + PREDICTORAS,
    "coeficiente": multiple.params.round(3).values,
    "error_estandar": multiple.bse.round(3).values,
    "estadistico_t": multiple.tvalues.round(2).values,
    "p_valor": [f"{p:.2e}" for p in multiple.pvalues],
    "ic95_inferior": ic_multiple[0].round(3).values,
    "ic95_superior": ic_multiple[1].round(3).values,
    "significativo": ["sí" if p < 0.05 else "no" for p in multiple.pvalues],
})
coef_multiple.to_csv(TABLAS / "regresion_multiple.csv", index=False)
print("\nCoeficientes del modelo múltiple")
print(coef_multiple.to_string(index=False))

# 4. Comparación de los dos modelos.
# El R² siempre sube al agregar variables; por eso se acompaña del R² ajustado,
# que penaliza los términos inútiles, y del RMSE, que está en millones de COP y
# se le puede explicar a alguien que no sepa estadística.


def resumen(modelo, nombre, formula):
    residuos = modelo.resid
    return {
        "modelo": nombre,
        "formula": formula,
        "variables": int(modelo.df_model),
        "r2": round(modelo.rsquared, 4),
        "r2_ajustado": round(modelo.rsquared_adj, 4),
        "rmse_millones": round(float(np.sqrt(np.mean(residuos ** 2))), 2),
        "mae_millones": round(float(np.mean(np.abs(residuos))), 2),
        "aic": round(modelo.aic, 1),
    }


comparacion = pd.DataFrame([
    resumen(simple, "Simple", "precio ~ area"),
    resumen(multiple, "Múltiple",
            "precio ~ area + habitaciones + antiguedad + estrato"),
])
comparacion.to_csv(TABLAS / "comparacion_modelos.csv", index=False)
print("\nComparación de modelos")
print(comparacion.to_string(index=False))

ganancia = (comparacion.loc[1, "r2"] - comparacion.loc[0, "r2"]) * 100
reduccion = (1 - comparacion.loc[1, "rmse_millones"]
             / comparacion.loc[0, "rmse_millones"]) * 100
print(f"\nEl modelo múltiple explica {ganancia:.1f} puntos porcentuales más de "
      f"la variabilidad del precio y reduce el error medio un {reduccion:.1f} %.")

# 5. Figuras con Matplotlib.

# Figura 1: la recta ajustada del modelo simple con su banda de confianza. La
# dispersión vertical alrededor de la recta es, visualmente, lo que las otras
# tres variables tendrán que explicar.
malla = np.linspace(df["area_m2"].min(), df["area_m2"].max(), 100)
prediccion = simple.get_prediction(
    pd.DataFrame({"area_m2": malla})).summary_frame()
b0, b1 = simple.params.iloc[0], simple.params.iloc[1]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.scatter(df["area_m2"], df[OBJETIVO], s=28, color=AZUL, edgecolor="white",
           linewidth=0.5, label="Apartamentos observados")
ax.plot(malla, prediccion["mean"], color=NARANJA, lw=2,
        label=f"Recta MCO: precio = {b0:.1f} + {b1:.2f} · área")
ax.fill_between(malla, prediccion["mean_ci_lower"], prediccion["mean_ci_upper"],
                color=NARANJA, alpha=0.25, label="Intervalo de confianza 95 %")
ax.set_title(f"Regresión simple: el área explica el {simple.rsquared:.1%} "
             "del precio")
ax.set_xlabel("Área (m²)")
ax.set_ylabel("Precio (millones de COP)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURAS / "ajuste_simple.png")
plt.close(fig)

# Figura 2: los errores de los dos modelos en la misma escala. Es la forma más
# honesta de mostrar la mejora: la nube del modelo múltiple es más estrecha.
fig, ejes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
limite = float(np.abs(simple.resid).max()) * 1.1
for ax, modelo, nombre in [(ejes[0], simple, "simple"),
                           (ejes[1], multiple, "múltiple")]:
    rmse = float(np.sqrt(np.mean(modelo.resid ** 2)))
    ax.scatter(modelo.fittedvalues, modelo.resid, s=26, color=AZUL,
               edgecolor="white", linewidth=0.4)
    ax.axhline(0, color=NARANJA, lw=1.4)
    ax.set_ylim(-limite, limite)
    ax.set_title(f"Modelo {nombre}  ·  R² = {modelo.rsquared:.4f}  ·  "
                 f"RMSE = {rmse:.1f}")
    ax.set_xlabel("Precio estimado (millones de COP)")
ejes[0].set_ylabel("Error de estimación (millones de COP)")
fig.suptitle("Los errores del modelo múltiple son la mitad de grandes",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURAS / "comparacion_residuos.png")
plt.close(fig)

# Figura 3: cuánto mueve el precio cada variable, con su intervalo de
# confianza. Es la figura que responde la pregunta de negocio.
bloque = coef_multiple.iloc[1:]
posicion = np.arange(len(bloque))[::-1]
error = np.vstack([
    bloque["coeficiente"] - bloque["ic95_inferior"],
    bloque["ic95_superior"] - bloque["coeficiente"],
])
colores = [AZUL if s == "sí" else GRIS for s in bloque["significativo"]]

fig, ax = plt.subplots(figsize=(7.8, 3.6))
ax.errorbar(bloque["coeficiente"], posicion, xerr=error, fmt="none",
            ecolor="#6baed6", elinewidth=2.5, capsize=5)
ax.scatter(bloque["coeficiente"], posicion, s=70, color=colores, zorder=3)
ax.axvline(0, color=NARANJA, lw=1.3, linestyle="--")
ax.set_ylim(-0.6, len(bloque) - 1 + 0.65)  # deja aire para las etiquetas
ax.set_yticks(posicion, ["Área\n(por m²)", "Habitaciones\n(por unidad)",
                         "Antigüedad\n(por año)", "Estrato\n(por nivel)"],
              fontsize=9)
for x, y in zip(bloque["coeficiente"], posicion):
    ax.annotate(f"{x:+.1f}", (x, y), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9,
                fontweight="bold")
ax.set_title("Efecto de cada variable sobre el precio "
             "(intervalo de confianza 95 %)")
ax.set_xlabel("Cambio en el precio (millones de COP)")
fig.tight_layout()
fig.savefig(FIGURAS / "efecto_variables.png")
plt.close(fig)

print("\nOK - Fase 2: 4 tablas en data/processed y 3 figuras en "
      "figures/python/regression")
