"""Actividad 4 - Fase 4: visualización avanzada con seaborn y Plotly.

Las fases anteriores dejaron el análisis resuelto; esta lo hace comunicable.
Se separa en dos bloques con propósitos distintos:

* **seaborn** produce las figuras estáticas de alta densidad que entran en el
  informe: matriz de dispersión, mapa de calor de correlaciones, ajustes por
  facetas y análisis de residuos con suavizado local.
* **Plotly** produce las piezas interactivas: un diagrama de dispersión
  explorable y un tablero de cuatro paneles con filtro por sector, que es el
  entregable navegable de la actividad.

Ambos bloques leen las tablas que escribieron las Fases 2 y 3, de modo que las
cifras mostradas son exactamente las estimadas allí y no un recálculo paralelo.

Rutas: el script se ubica en codes -> utils -> raíz del proyecto.
Escribe las figuras estáticas en
``public/assets/images/figures/python/advanced/`` y las interactivas en
``public/assets/images/figures/python/dashboard/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_BASE = PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python"
ADVANCED_DIR = FIGURES_BASE / "advanced"
DASHBOARD_DIR = FIGURES_BASE / "dashboard"
for d in (ADVANCED_DIR, DASHBOARD_DIR):
    d.mkdir(parents=True, exist_ok=True)

SECTOR_ORDER = ["Residencial", "Comercial", "Industrial"]
SECTOR_COLORS = {"Residencial": "#a6bddb", "Comercial": "#74a9cf",
                 "Industrial": "#2b8cbe"}
ACCENT = "#d95f02"

sns.set_theme(style="whitegrid", palette=list(SECTOR_COLORS.values()),
              rc={"figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
                  "axes.titleweight": "bold", "grid.alpha": 0.3})

df = pd.read_csv(DATA_DIR / "consumo_energia.csv")
df["sector"] = pd.Categorical(df["sector"], categories=SECTOR_ORDER, ordered=True)
df["tarifa_cop_kwh"] = df["costo_miles_cop"] * 1000 / df["consumo_kwh"]

REF = "C(sector, Treatment(reference='Residencial'))"
m1 = smf.ols("costo_miles_cop ~ consumo_kwh", data=df).fit()
m3 = smf.ols(f"costo_miles_cop ~ consumo_kwh * {REF}", data=df).fit()
df["ajustado_m1"] = m1.fittedvalues
df["residuo_m1"] = m1.resid
df["ajustado_m3"] = m3.fittedvalues
df["residuo_m3"] = m3.resid

comparacion = pd.read_csv(PROCESSED_DIR / "comparacion_modelos.csv")
coeficientes = pd.read_csv(PROCESSED_DIR / "regresion_multiple.csv")
tarifas = pd.read_csv(PROCESSED_DIR / "tarifas_estimadas.csv")

"""1. MATRIZ DE DISPERSIÓN (seaborn).

Un ``pairplot`` cruza todas las variables continuas a la vez y pone la
densidad de cada una en la diagonal. Es la vista que justifica de un golpe por
qué el sector debía entrar al modelo: los tres grupos ocupan regiones
prácticamente disjuntas en cualquiera de los planos.
"""
variables = ["consumo_kwh", "costo_miles_cop", "tarifa_cop_kwh"]
rejilla = sns.pairplot(df, vars=variables, hue="sector", diag_kind="kde",
                       plot_kws=dict(s=26, edgecolor="white", linewidth=0.4),
                       height=2.3, corner=True)
etiquetas = {"consumo_kwh": "Consumo (kWh/mes)",
             "costo_miles_cop": "Costo (miles COP)",
             "tarifa_cop_kwh": "Tarifa (COP/kWh)"}
for i, fila in enumerate(variables):
    for j, col in enumerate(variables):
        ax = rejilla.axes[i][j]
        if ax is None:
            continue
        ax.set_xlabel(etiquetas[col], fontsize=9)
        ax.set_ylabel(etiquetas[fila], fontsize=9)
rejilla.figure.suptitle("Matriz de dispersión por sector: tres poblaciones "
                        "casi disjuntas", y=1.02, fontsize=12,
                        fontweight="bold")
rejilla.savefig(ADVANCED_DIR / "sns_matriz_dispersion.png",
                bbox_inches="tight")
plt.close(rejilla.figure)

"""2. MAPA DE CALOR DE CORRELACIONES (seaborn).

El triángulo inferior evita repetir la información simétrica y la paleta
divergente centrada en cero distingue de un vistazo asociaciones positivas de
negativas: la tarifa cae cuando el consumo sube, que es el descuento por
escala que después confirma la regresión.
"""
matriz = df[["consumo_kwh", "costo_miles_cop", "tarifa_cop_kwh"]].corr()
mascara = np.triu(np.ones_like(matriz, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(5.8, 4.6))
sns.heatmap(matriz, mask=mascara, annot=True, fmt=".3f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, linewidths=0.6, square=True,
            cbar_kws={"label": "Coeficiente de Pearson"}, ax=ax)
ax.set_title("Correlaciones entre las variables del modelo")
ax.set_xticklabels(["Consumo", "Costo", "Tarifa"], rotation=0, fontsize=9)
ax.set_yticklabels(["Consumo", "Costo", "Tarifa"], rotation=0, fontsize=9)
fig.tight_layout()
fig.savefig(ADVANCED_DIR / "sns_heatmap_correlacion.png")
plt.close(fig)

"""3. AJUSTE POR FACETAS (seaborn).

``lmplot`` ajusta y dibuja una regresión independiente dentro de cada faceta
con su banda de confianza. Separar los sectores en paneles con ejes libres es
lo que permite ver la relación dentro de cada grupo sin que la escala del
sector Industrial aplaste a los otros dos.
"""
facetas = sns.lmplot(data=df, x="consumo_kwh", y="costo_miles_cop",
                     col="sector", hue="sector", height=3.2, aspect=1.0,
                     facet_kws=dict(sharex=False, sharey=False),
                     scatter_kws=dict(s=28, edgecolor="white"),
                     line_kws=dict(color=ACCENT, linewidth=1.8), ci=95)
for ax, s in zip(facetas.axes.flat, SECTOR_ORDER):
    tarifa = tarifas.loc[tarifas["sector"] == s,
                         "tarifa_implicita_cop_kwh"].iloc[0]
    ax.set_title(f"{s} · {tarifa:,.0f} COP/kWh", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("Consumo (kWh/mes)")
    ax.set_ylabel("Costo (miles de COP)")
facetas.figure.suptitle("Una regresión por sector, cada una en su propia "
                        "escala", y=1.04, fontsize=12, fontweight="bold")
facetas.savefig(ADVANCED_DIR / "sns_lmplot_sectores.png", bbox_inches="tight")
plt.close(facetas.figure)

"""4. RESIDUOS CON SUAVIZADO LOCAL (seaborn).

``residplot`` con ``lowess`` traza una curva no paramétrica sobre los
residuos: si el modelo capturó toda la estructura, esa curva debe ser plana.
Comparar M1 y M3 en la misma figura muestra el efecto de añadir el sector.
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
for ax, columna, titulo in [
    (ax1, "residuo_m1", "M1 · simple: la curva se desvía de cero"),
    (ax2, "residuo_m3", "M3 · con interacción: curva plana"),
]:
    sns.residplot(data=df, x=columna.replace("residuo", "ajustado"), y=columna,
                  lowess=True, ax=ax, scatter_kws=dict(s=22, alpha=0.7,
                                                       color="#2c7fb8"),
                  line_kws=dict(color=ACCENT, linewidth=1.8))
    ax.axhline(0, color="#636363", lw=1.0, linestyle=":")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Valor ajustado (miles de COP)")
    ax.set_ylabel("Residuo (miles de COP)")
fig.suptitle("Residuos con suavizado local: qué estructura queda sin explicar",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(ADVANCED_DIR / "sns_residuos_lowess.png")
plt.close(fig)

"""5. DISPERSIÓN INTERACTIVA (Plotly).

La misma información del gráfico estático, más lo que este no puede dar: al
posar el cursor sobre un punto aparece el identificador del cliente y sus
cifras exactas, y la leyenda permite aislar un sector con un clic.
"""
figura_scatter = px.scatter(
    df, x="consumo_kwh", y="costo_miles_cop", color="sector",
    trendline="ols", trendline_scope="trace",
    color_discrete_map=SECTOR_COLORS,
    category_orders={"sector": SECTOR_ORDER},
    hover_data={"cliente_id": True, "tarifa_cop_kwh": ":.1f",
                "consumo_kwh": ":.1f", "costo_miles_cop": ":.1f"},
    labels={"consumo_kwh": "Consumo (kWh/mes)",
            "costo_miles_cop": "Costo facturado (miles de COP)",
            "sector": "Sector", "tarifa_cop_kwh": "Tarifa (COP/kWh)"},
    title="Regresión por sector explorable cliente a cliente")
figura_scatter.update_traces(marker=dict(size=8, line=dict(width=0.6,
                                                           color="white")))
figura_scatter.update_layout(template="plotly_white", width=950, height=560,
                             legend=dict(orientation="h", y=1.02, x=0))
figura_scatter.write_html(DASHBOARD_DIR / "scatter_interactivo.html",
                          include_plotlyjs="cdn")
figura_scatter.write_image(DASHBOARD_DIR / "scatter_interactivo.png", scale=2)

"""6. TABLERO INTERACTIVO DE CUATRO PANELES (Plotly).

El tablero reúne en una sola página las cuatro preguntas del análisis: cómo
ajusta el modelo, qué queda en los residuos, cuál de las especificaciones
gana y qué coeficientes son distintos de cero. Los botones superiores filtran
los tres primeros paneles por sector, de modo que el lector puede aislar un
grupo sin regenerar nada.
"""
tablero = make_subplots(
    rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.10,
    subplot_titles=(
        "Ajuste del modelo M3 por sector",
        "Residuos frente a valores ajustados",
        "RMSE de los tres modelos (miles de COP)",
        "Coeficientes de M3 con intervalo al 95 %"))

trazas_por_sector = []
for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    tabla = pd.DataFrame({"consumo_kwh": np.linspace(sub["consumo_kwh"].min(),
                                                     sub["consumo_kwh"].max(), 40),
                          "sector": s})
    tablero.add_trace(go.Scatter(
        x=sub["consumo_kwh"], y=sub["costo_miles_cop"], mode="markers",
        name=s, legendgroup=s, marker=dict(size=7, color=SECTOR_COLORS[s],
                                           line=dict(width=0.5, color="white")),
        text=sub["cliente_id"],
        hovertemplate="%{text}<br>%{x:.1f} kWh<br>%{y:.1f} miles COP"
                      "<extra></extra>"), row=1, col=1)
    trazas_por_sector.append(s)
    tablero.add_trace(go.Scatter(
        x=tabla["consumo_kwh"], y=m3.predict(tabla), mode="lines",
        name=f"Ajuste {s}", legendgroup=s, showlegend=False,
        line=dict(color=SECTOR_COLORS[s], width=3)), row=1, col=1)
    trazas_por_sector.append(s)
    tablero.add_trace(go.Scatter(
        x=sub["ajustado_m3"], y=sub["residuo_m3"], mode="markers",
        name=s, legendgroup=s, showlegend=False,
        marker=dict(size=7, color=SECTOR_COLORS[s],
                    line=dict(width=0.5, color="white")),
        text=sub["cliente_id"],
        hovertemplate="%{text}<br>ajustado %{x:.1f}<br>residuo %{y:.1f}"
                      "<extra></extra>"), row=1, col=2)
    trazas_por_sector.append(s)

tablero.add_hline(y=0, line=dict(color=ACCENT, width=1.5), row=1, col=2)

tablero.add_trace(go.Bar(
    x=["M1", "M2", "M3"], y=comparacion["rmse"],
    marker_color=["#bdd7e7", "#6baed6", "#2b8cbe"],
    text=comparacion["rmse"].round(1), textposition="outside",
    name="RMSE", showlegend=False,
    customdata=comparacion["especificacion"],
    hovertemplate="%{x}: %{customdata}<br>RMSE %{y:.2f}<extra></extra>"),
    row=2, col=1)
trazas_por_sector.append(None)

pendientes = coeficientes.iloc[3:].copy()
pendientes["etiqueta"] = ["Pendiente Residencial", "Δ pendiente Comercial",
                          "Δ pendiente Industrial"]
tablero.add_trace(go.Scatter(
    x=pendientes["coeficiente"], y=pendientes["etiqueta"], mode="markers",
    marker=dict(size=11, color="#2b8cbe"), name="Coeficiente",
    showlegend=False,
    error_x=dict(type="data", symmetric=False,
                 array=pendientes["ic95_superior"] - pendientes["coeficiente"],
                 arrayminus=pendientes["coeficiente"] - pendientes["ic95_inferior"],
                 color="#6baed6", thickness=2, width=6),
    hovertemplate="%{y}<br>coeficiente %{x:.4f}<extra></extra>"), row=2, col=2)
trazas_por_sector.append(None)
tablero.add_vline(x=0, line=dict(color=ACCENT, width=1.5, dash="dash"),
                  row=2, col=2)

botones = [dict(label="Todos los sectores", method="update",
                args=[{"visible": [True] * len(trazas_por_sector)}])]
for s in SECTOR_ORDER:
    botones.append(dict(
        label=s, method="update",
        args=[{"visible": [t is None or t == s for t in trazas_por_sector]}]))

tablero.update_xaxes(title_text="Consumo (kWh/mes)", row=1, col=1)
tablero.update_yaxes(title_text="Costo (miles de COP)", row=1, col=1)
tablero.update_xaxes(title_text="Valor ajustado (miles de COP)", row=1, col=2)
tablero.update_yaxes(title_text="Residuo (miles de COP)", row=1, col=2)
tablero.update_xaxes(title_text="Modelo", row=2, col=1)
tablero.update_yaxes(title_text="RMSE (miles de COP)", row=2, col=1)
tablero.update_xaxes(title_text="Miles de COP por kWh", row=2, col=2)
tablero.update_yaxes(tickfont=dict(size=9), row=2, col=2)
tablero.update_layout(
    template="plotly_white", width=1200, height=830,
    margin=dict(t=170, l=90, r=40, b=60),
    title=dict(text="Tablero de regresión · consumo energético por sector",
               font=dict(size=18), x=0, xanchor="left", y=0.97),
    legend=dict(orientation="h", y=1.135, x=0.42, yanchor="middle"),
    updatemenus=[dict(type="buttons", direction="right", x=0, y=1.14,
                      xanchor="left", buttons=botones, showactive=True,
                      bgcolor="#f0f0f0")])
tablero.write_html(DASHBOARD_DIR / "dashboard_regresion.html",
                   include_plotlyjs="cdn")
tablero.write_image(DASHBOARD_DIR / "dashboard_regresion.png", scale=2)

print("\nOK - Fase 4: figuras de seaborn y tablero interactivo generados")
