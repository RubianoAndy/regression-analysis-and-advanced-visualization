"""Actividad 4 - Fase 1: correlación y regresión lineal simple con statsmodels.

Lee el dataset de consumo energético heredado de las actividades anteriores,
cuantifica la asociación entre el consumo (kWh) y el costo facturado
(miles de COP) mediante los coeficientes de Pearson y Spearman, y ajusta por
mínimos cuadrados ordinarios el modelo de regresión lineal simple
``costo = b0 + b1 * consumo``.

El interés del ajuste no está solo en su bondad global, sino en lo que dejan
ver sus residuos: al colorearlos por sector aparece un patrón sistemático que
la especificación simple no puede capturar y que motiva la Fase 2.

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
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
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

"""Dataset: las mismas 120 observaciones de las actividades anteriores
(semilla fija 42), de modo que el foco esté en el modelo y no en los datos."""
dataset_path = DATA_DIR / "consumo_energia.csv"
if not dataset_path.exists():
    raise SystemExit(
        f"No se encontró {dataset_path}. Ejecuta antes la Fase 0: "
        "python utils/codes/dataset.py"
    )
df = pd.read_csv(dataset_path)
df["sector"] = pd.Categorical(df["sector"], categories=SECTOR_ORDER, ordered=True)
df["tarifa_cop_kwh"] = df["costo_miles_cop"] * 1000 / df["consumo_kwh"]
n = len(df)

"""1. ANÁLISIS DE CORRELACIÓN.

Antes de ajustar cualquier recta conviene medir si existe asociación y de qué
tipo. Pearson mide la componente lineal y Spearman la monótona; su cercanía
indica que la relación no solo es creciente sino además aproximadamente
lineal. El cálculo se repite dentro de cada sector para comprobar que la
asociación no es un artefacto de mezclar tres poblaciones de escalas
distintas (paradoja de Simpson).
"""
corr_rows = []
for label in ["Global"] + SECTOR_ORDER:
    sub = df if label == "Global" else df[df["sector"] == label]
    r, p_r = stats.pearsonr(sub["consumo_kwh"], sub["costo_miles_cop"])
    rho, p_rho = stats.spearmanr(sub["consumo_kwh"], sub["costo_miles_cop"])
    corr_rows.append({
        "grupo": label,
        "n": len(sub),
        "pearson_r": round(r, 4),
        "pearson_p": f"{p_r:.2e}",
        "r_cuadrado": round(r ** 2, 4),
        "spearman_rho": round(rho, 4),
        "spearman_p": f"{p_rho:.2e}",
        "tarifa_media_cop_kwh": round(sub["tarifa_cop_kwh"].mean(), 1),
    })
correlaciones = pd.DataFrame(corr_rows)
correlaciones.to_csv(PROCESSED_DIR / "correlaciones.csv", index=False)
print("Correlación entre consumo (kWh) y costo (miles COP)")
print(correlaciones.to_string(index=False))

"""2. REGRESIÓN LINEAL SIMPLE POR MCO.

Se estima costo = b0 + b1 * consumo. La pendiente b1 tiene lectura directa:
es la tarifa media implícita en COP por kWh (el costo está en miles de COP,
así que b1 * 1000 devuelve la tarifa). El intercepto b0 representa el costo
esperado con consumo nulo, un valor sin sentido físico aquí porque cae fuera
del recorrido observado (121,2 kWh es el mínimo).
"""
modelo_simple = smf.ols("costo_miles_cop ~ consumo_kwh", data=df).fit()
print("\n" + "=" * 72)
print(modelo_simple.summary())

conf = modelo_simple.conf_int(alpha=0.05)
coef_simple = pd.DataFrame({
    "termino": ["Intercepto (b0)", "consumo_kwh (b1)"],
    "coeficiente": modelo_simple.params.round(4).values,
    "error_std": modelo_simple.bse.round(4).values,
    "estadistico_t": modelo_simple.tvalues.round(2).values,
    "p_valor": [f"{p:.2e}" for p in modelo_simple.pvalues],
    "ic95_inferior": conf[0].round(4).values,
    "ic95_superior": conf[1].round(4).values,
})
coef_simple.to_csv(PROCESSED_DIR / "regresion_simple.csv", index=False)
print("\nCoeficientes del modelo simple")
print(coef_simple.to_string(index=False))

b0, b1 = modelo_simple.params
residuos = modelo_simple.resid
ajustados = modelo_simple.fittedvalues
rmse = float(np.sqrt(np.mean(residuos ** 2)))
mae = float(np.mean(np.abs(residuos)))

"""3. DIAGNÓSTICO DE LOS SUPUESTOS DEL MODELO.

Un R2 alto no valida un modelo: los cuatro supuestos de MCO se contrastan por
separado. Breusch-Pagan evalúa homocedasticidad (H0: varianza constante),
Jarque-Bera la normalidad de los residuos, Durbin-Watson su independencia
(valores próximos a 2 la respaldan) y el RESET de Ramsey la especificación
funcional (H0: la forma lineal es adecuada).
"""
bp_stat, bp_p, _, _ = het_breuschpagan(residuos, modelo_simple.model.exog)
jb_stat, jb_p, _, _ = jarque_bera(residuos)
dw = durbin_watson(residuos)
reset = sm.stats.diagnostic.linear_reset(modelo_simple, power=2, use_f=True)

diagnostico_simple = pd.DataFrame([
    {"prueba": "Breusch-Pagan (homocedasticidad)", "estadistico": round(bp_stat, 3),
     "p_valor": f"{bp_p:.2e}", "supuesto": "Varianza constante",
     "conclusion": "Se rechaza" if bp_p < 0.05 else "No se rechaza"},
    {"prueba": "Jarque-Bera (normalidad)", "estadistico": round(jb_stat, 3),
     "p_valor": f"{jb_p:.2e}", "supuesto": "Residuos normales",
     "conclusion": "Se rechaza" if jb_p < 0.05 else "No se rechaza"},
    {"prueba": "Durbin-Watson (independencia)", "estadistico": round(dw, 3),
     "p_valor": "-", "supuesto": "Residuos no correlacionados",
     "conclusion": "No se rechaza" if 1.5 < dw < 2.5 else "Se rechaza"},
    {"prueba": "RESET de Ramsey (especificación)",
     "estadistico": round(float(reset.statistic), 3),
     "p_valor": f"{float(reset.pvalue):.2e}", "supuesto": "Forma lineal adecuada",
     "conclusion": "Se rechaza" if float(reset.pvalue) < 0.05 else "No se rechaza"},
])
diagnostico_simple.to_csv(PROCESSED_DIR / "diagnostico_simple.csv", index=False)
print("\nDiagnóstico de supuestos del modelo simple")
print(diagnostico_simple.to_string(index=False))

"""El sesgo por sector es el hallazgo que abre la Fase 2: si el modelo fuera
correcto, la media de los residuos sería nula dentro de cualquier subgrupo."""
sesgo = (df.assign(residuo=residuos)
           .groupby("sector", observed=True)["residuo"]
           .agg(n="count", sesgo_medio="mean", desv_std="std")
           .round(2)
           .reset_index())
sesgo.to_csv(PROCESSED_DIR / "sesgo_por_sector.csv", index=False)
print("\nSesgo medio de los residuos por sector (miles de COP)")
print(sesgo.to_string(index=False))
print(f"\nRMSE = {rmse:,.2f} miles COP | MAE = {mae:,.2f} miles COP")
print(f"Tarifa implícita en la pendiente: {b1 * 1000:,.1f} COP/kWh")

"""4. FIGURAS DE LA REGRESIÓN SIMPLE.

Cada figura responde una pregunta: si hay relación y cuánta incertidumbre
tiene la recta, si se cumplen los supuestos y dónde falla el modelo.
"""

"""Diagrama de dispersión con la recta ajustada, su ecuación y las dos bandas
de incertidumbre: la estrecha acota dónde está la recta media de la población
y la ancha dónde caerá una factura individual. El color queda reservado para
el sector, que todavía no forma parte del modelo."""
grid = np.linspace(df["consumo_kwh"].min(), df["consumo_kwh"].max(), 200)
pred = modelo_simple.get_prediction(
    pd.DataFrame({"consumo_kwh": grid})).summary_frame(alpha=0.05)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.fill_between(grid, pred["obs_ci_lower"], pred["obs_ci_upper"],
                color="#2b8cbe", alpha=0.12,
                label="Intervalo de predicción 95 % (una factura)")
ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"],
                color=ACCENT, alpha=0.30,
                label="Intervalo de confianza 95 % (recta media)")
for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    ax.scatter(sub["consumo_kwh"], sub["costo_miles_cop"], s=26,
               color=SECTOR_COLORS[s], edgecolor="white", linewidth=0.5, label=s)
ax.plot(grid, pred["mean"], color=ACCENT, lw=1.8,
        label=f"MCO: ŷ = {b0:.2f} + {b1:.4f}·x")
ax.set_title(f"Relación consumo–costo: r = {correlaciones.loc[0, 'pearson_r']:.4f}, "
             f"R² = {modelo_simple.rsquared:.4f}")
ax.set_xlabel("Consumo (kWh/mes)")
ax.set_ylabel("Costo facturado (miles de COP)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "dispersion_ajuste_simple.png")
plt.close(fig)

"""Panel de diagnóstico en cuatro vistas. El color por sector convierte un
gráfico de control rutinario en el argumento central de la actividad: los
residuos no son ruido, están ordenados por una variable omitida."""
fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6))

ax = axes[0, 0]
for s in SECTOR_ORDER:
    m = (df["sector"] == s).to_numpy()
    ax.scatter(ajustados[m], residuos[m], s=22, color=SECTOR_COLORS[s],
               edgecolor="white", linewidth=0.4, label=s)
ax.axhline(0, color=ACCENT, lw=1.2)
ax.set_title("Residuos frente a valores ajustados")
ax.set_xlabel("Costo ajustado (miles de COP)")
ax.set_ylabel("Residuo (miles de COP)")
ax.legend(fontsize=7)

ax = axes[0, 1]
sm.qqplot(residuos, line="s", ax=ax, markerfacecolor="#2c7fb8",
          markeredgecolor="white", markersize=4, alpha=0.9)
ax.get_lines()[1].set_color(ACCENT)
ax.set_title(f"Gráfico Q-Q normal (Jarque-Bera p = {jb_p:.2e})")
ax.set_xlabel("Cuantiles teóricos")
ax.set_ylabel("Cuantiles de los residuos")

ax = axes[1, 0]
raiz_std = np.sqrt(np.abs(residuos / residuos.std(ddof=1)))
ax.scatter(ajustados, raiz_std, s=22, color="#2c7fb8", edgecolor="white",
           linewidth=0.4)
coef_ls = np.polyfit(ajustados, raiz_std, 1)
ax.plot(np.sort(ajustados), np.polyval(coef_ls, np.sort(ajustados)),
        color=ACCENT, lw=1.4)
ax.set_title(f"Escala-localización (Breusch-Pagan p = {bp_p:.2e})")
ax.set_xlabel("Costo ajustado (miles de COP)")
ax.set_ylabel("√|residuo estandarizado|")

ax = axes[1, 1]
influencia = modelo_simple.get_influence()
cook = influencia.cooks_distance[0]
ax.stem(np.arange(n), cook, linefmt="#2c7fb8", markerfmt=" ", basefmt=" ")
umbral = 4 / n
ax.axhline(umbral, color=ACCENT, lw=1.2, linestyle="--",
           label=f"Umbral 4/n = {umbral:.3f}")
ax.set_title(f"Influencia: {int((cook > umbral).sum())} puntos sobre el umbral")
ax.set_xlabel("Índice de la observación")
ax.set_ylabel("Distancia de Cook")
ax.legend(fontsize=7)

fig.suptitle("Diagnóstico del modelo simple: los residuos se ordenan por sector",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "diagnostico_simple.png")
plt.close(fig)

"""Residuos por sector: el sesgo sistemático, ya sin la dispersión del
gráfico anterior. Cada caja debería centrarse en cero y ninguna lo hace."""
fig, ax = plt.subplots(figsize=(7.0, 3.8))
datos = [residuos[(df["sector"] == s).to_numpy()] for s in SECTOR_ORDER]
bp = ax.boxplot(datos, tick_labels=SECTOR_ORDER, patch_artist=True,
                medianprops=dict(color="black"))
for patch, s in zip(bp["boxes"], SECTOR_ORDER):
    patch.set_facecolor(SECTOR_COLORS[s])
ax.axhline(0, color=ACCENT, lw=1.2, label="Residuo nulo (modelo insesgado)")
tope = float(residuos.max())
ax.set_ylim(float(residuos.min()) * 1.15, tope * 1.35)
for i, s in enumerate(SECTOR_ORDER):
    media = sesgo.loc[sesgo["sector"] == s, "sesgo_medio"].iloc[0]
    ax.scatter(i + 1, media, color=ACCENT, marker="D", s=26, zorder=3)
    ax.annotate(f"media = {media:+.1f}", (i + 1, tope * 1.18),
                ha="center", fontsize=8, color=ACCENT)
ax.set_title("Sesgo sistemático del modelo simple: el sector explica el residuo")
ax.set_xlabel("Sector")
ax.set_ylabel("Residuo (miles de COP)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "residuos_por_sector.png")
plt.close(fig)

print("\nOK - Fase 1: correlación, regresión simple y diagnóstico generados")
