"""Fase 4 - Visualización avanzada con seaborn y Plotly.

Las fases anteriores dejaron el análisis resuelto; esta lo hace comunicable.

* **seaborn** produce las tres figuras estáticas del informe: matriz de
  dispersión, mapa de calor de correlaciones y ajuste por estrato.
* **Plotly** produce el tablero interactivo: cuatro paneles con zoom, valores
  al pasar el cursor y un filtro por estrato.

Ejecutar desde la raíz del proyecto:
    python utils/codes/python/visualization.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "data" / "dataset" / "viviendas.csv"
TABLAS = PROJECT_ROOT / "data" / "processed"
BASE = PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python"
AVANZADAS = BASE / "advanced"
TABLERO = BASE / "dashboard"
for carpeta in (AVANZADAS, TABLERO):
    carpeta.mkdir(parents=True, exist_ok=True)

COLORES_ESTRATO = {3: "#a6bddb", 4: "#4292c6", 5: "#08519c"}
NARANJA = "#d95f02"

sns.set_theme(style="whitegrid",
              rc={"figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
                  "axes.titleweight": "bold", "grid.alpha": 0.3})

df = pd.read_csv(DATASET)
coeficientes = pd.read_csv(TABLAS / "regresion_multiple.csv")

# El modelo múltiple de la Fase 2, reajustado aquí para poder graficar sus
# predicciones sin arrastrar objetos entre scripts.
multiple = smf.ols(
    "precio_millones_cop ~ area_m2 + habitaciones + antiguedad_anios + estrato",
    data=df,
).fit()
df["precio_estimado"] = multiple.fittedvalues.round(1)

ETIQUETAS = {
    "area_m2": "Área (m²)",
    "habitaciones": "Habitaciones",
    "antiguedad_anios": "Antigüedad (años)",
    "estrato": "Estrato",
    "precio_millones_cop": "Precio (millones COP)",
}

# BLOQUE 1 - Figuras estáticas con seaborn

# Figura 1: matriz de dispersión. Cruza todas las variables a la vez y colorea
# por estrato; de un vistazo se ve que el estrato separa el precio en bandas.
variables = ["area_m2", "antiguedad_anios", "precio_millones_cop"]
rejilla = sns.pairplot(df, vars=variables, hue="estrato", corner=True,
                       diag_kind="kde", height=2.3,
                       palette=COLORES_ESTRATO,
                       plot_kws=dict(s=28, edgecolor="white", linewidth=0.4))
for i, fila in enumerate(variables):
    for j, columna in enumerate(variables):
        eje = rejilla.axes[i][j]
        if eje is not None:
            eje.set_xlabel(ETIQUETAS[columna], fontsize=9)
            eje.set_ylabel(ETIQUETAS[fila], fontsize=9)
rejilla.figure.suptitle("Matriz de dispersión: el estrato ordena el precio en "
                        "bandas", y=1.02, fontsize=12, fontweight="bold")
rejilla.savefig(AVANZADAS / "sns_matriz_dispersion.png", bbox_inches="tight")
plt.close(rejilla.figure)

# Figura 2: mapa de calor de correlaciones. El triángulo inferior evita
# repetir información y la paleta divergente distingue el signo: la antigüedad
# es la única variable que empuja el precio hacia abajo.
matriz = df[list(ETIQUETAS)].corr()
mascara = np.triu(np.ones_like(matriz, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(6.4, 5.0))
sns.heatmap(matriz, mask=mascara, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, linewidths=0.6, square=True,
            cbar_kws={"label": "Coeficiente de Pearson"}, ax=ax)
etiquetas_cortas = ["Área", "Habitac.", "Antigüedad", "Estrato", "Precio"]
ax.set_xticklabels(etiquetas_cortas, rotation=0, fontsize=9)
ax.set_yticklabels(etiquetas_cortas, rotation=0, fontsize=9)
ax.set_title("Correlación entre las variables del modelo")
fig.tight_layout()
fig.savefig(AVANZADAS / "sns_heatmap_correlacion.png")
plt.close(fig)

# Figura 3: una regresión por estrato. Las tres rectas son casi paralelas, que
# es exactamente el supuesto del modelo múltiple: el estrato desplaza el precio
# hacia arriba sin cambiar cuánto vale el metro cuadrado.
rejilla = sns.lmplot(data=df, x="area_m2", y="precio_millones_cop",
                     hue="estrato", palette=COLORES_ESTRATO, height=4.2,
                     aspect=1.5, ci=95,
                     scatter_kws=dict(s=30, edgecolor="white"))
rejilla.set_axis_labels(ETIQUETAS["area_m2"], ETIQUETAS["precio_millones_cop"])
rejilla.figure.suptitle("Tres rectas casi paralelas: el estrato sube el precio, "
                        "no el valor del m²", y=1.02, fontsize=12,
                        fontweight="bold")
rejilla.savefig(AVANZADAS / "sns_ajuste_por_estrato.png", bbox_inches="tight")
plt.close(rejilla.figure)

print("OK - seaborn: 3 figuras en figures/python/advanced")

# BLOQUE 2 - Piezas interactivas con Plotly

# Pieza 1: dispersión interactiva con línea de tendencia MCO. Al pasar el
# cursor muestra la ficha de cada apartamento, algo imposible en una imagen.
dispersion = px.scatter(
    df, x="area_m2", y="precio_millones_cop", color="estrato",
    size="habitaciones", trendline="ols", trendline_scope="overall",
    trendline_color_override=NARANJA,
    color_continuous_scale=["#a6bddb", "#4292c6", "#08519c"],
    hover_data={"inmueble_id": True, "antiguedad_anios": True,
                "habitaciones": True},
    labels=ETIQUETAS,
    title="Precio frente a área · el tamaño del punto es el número de habitaciones",
)
dispersion.update_layout(template="plotly_white", height=520,
                         title_font_size=15)
dispersion.write_html(TABLERO / "dispersion_interactiva.html",
                      include_plotlyjs="cdn")

# Pieza 2: tablero de cuatro paneles con filtro por estrato.
estratos = sorted(df["estrato"].unique())
tablero = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "1 · Precio frente a área, por estrato",
        "2 · Precio promedio por estrato",
        "3 · Precio real frente al estimado por el modelo",
        "4 · Cuánto suma o resta cada variable al precio",
    ),
)

# Panel 1 y panel 3: una traza por estrato, para poder filtrarlas después.
for estrato in estratos:
    sub = df[df["estrato"] == estrato]
    tablero.add_trace(go.Scatter(
        x=sub["area_m2"], y=sub["precio_millones_cop"], mode="markers",
        name=f"Estrato {estrato}", legendgroup=f"e{estrato}",
        marker=dict(size=8, color=COLORES_ESTRATO[estrato],
                    line=dict(width=0.5, color="white")),
        customdata=sub[["inmueble_id", "antiguedad_anios", "habitaciones"]],
        hovertemplate=("<b>%{customdata[0]}</b><br>Área: %{x:.0f} m²<br>"
                       "Precio: %{y:.0f} M COP<br>"
                       "Antigüedad: %{customdata[1]} años<br>"
                       "Habitaciones: %{customdata[2]}<extra></extra>"),
    ), row=1, col=1)

promedios = df.groupby("estrato")["precio_millones_cop"].mean().round(1)
tablero.add_trace(go.Bar(
    x=[f"Estrato {e}" for e in promedios.index], y=promedios.values,
    marker_color=[COLORES_ESTRATO[e] for e in promedios.index],
    text=promedios.values, textposition="outside", showlegend=False,
    hovertemplate="%{x}<br>Precio promedio: %{y:.0f} M COP<extra></extra>",
), row=1, col=2)

for estrato in estratos:
    sub = df[df["estrato"] == estrato]
    tablero.add_trace(go.Scatter(
        x=sub["precio_estimado"], y=sub["precio_millones_cop"], mode="markers",
        name=f"Estrato {estrato}", legendgroup=f"e{estrato}", showlegend=False,
        marker=dict(size=8, color=COLORES_ESTRATO[estrato],
                    line=dict(width=0.5, color="white")),
        hovertemplate=("Estimado: %{x:.0f} M COP<br>"
                       "Real: %{y:.0f} M COP<extra></extra>"),
    ), row=2, col=1)

diagonal = [df["precio_millones_cop"].min(), df["precio_millones_cop"].max()]
tablero.add_trace(go.Scatter(
    x=diagonal, y=diagonal, mode="lines", name="Predicción perfecta",
    line=dict(color=NARANJA, dash="dash", width=2), showlegend=False,
    hoverinfo="skip",
), row=2, col=1)

efectos = coeficientes[coeficientes["termino"] != "Intercepto"]
nombres = ["Área<br>(+1 m²)", "Habitaciones<br>(+1)", "Antigüedad<br>(+1 año)",
           "Estrato<br>(+1 nivel)"]
tablero.add_trace(go.Bar(
    x=nombres, y=efectos["coeficiente"],
    marker_color=["#08519c" if v > 0 else NARANJA
                  for v in efectos["coeficiente"]],
    text=[f"{v:+.1f}" for v in efectos["coeficiente"]], textposition="outside",
    showlegend=False,
    hovertemplate="%{x}<br>Efecto: %{y:+.2f} M COP<extra></extra>",
), row=2, col=2)

# El filtro apaga y enciende las trazas de estrato de los paneles 1 y 3; los
# paneles 2 y 4 resumen el modelo completo y se mantienen siempre visibles.
n_estratos = len(estratos)
FIJAS = [True]  # panel 2
DIAGONAL_Y_EFECTOS = [True, True]  # diagonal del panel 3 y panel 4


def visibilidad(seleccion=None):
    marcas = [seleccion is None or e == seleccion for e in estratos]
    return marcas + FIJAS + marcas + DIAGONAL_Y_EFECTOS


tablero.update_layout(
    template="plotly_white", height=860, width=1200,
    title=dict(text="<b>Tablero de precios de vivienda usada en Bogotá</b><br>"
                    f"<sub>{len(df)} apartamentos · modelo múltiple con "
                    f"R² = {multiple.rsquared:.3f} · use el menú para filtrar "
                    "por estrato</sub>",
               x=0.5, xanchor="center", font_size=17),
    legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center",
                x=0.5),
    margin=dict(t=130, b=150),
    updatemenus=[dict(
        type="dropdown", direction="down", x=0.01, y=1.14, xanchor="left",
        showactive=True, bgcolor="white", bordercolor="#cccccc",
        buttons=[dict(label="Todos los estratos", method="update",
                      args=[{"visible": visibilidad()}])]
                + [dict(label=f"Solo estrato {e}", method="update",
                        args=[{"visible": visibilidad(e)}]) for e in estratos],
    )],
)
tablero.update_xaxes(title_text="Área (m²)", row=1, col=1)
tablero.update_yaxes(title_text="Precio (millones COP)", row=1, col=1)
tablero.update_yaxes(title_text="Precio promedio (millones COP)", row=1, col=2)
tablero.update_xaxes(title_text="Precio estimado (millones COP)", row=2, col=1)
tablero.update_yaxes(title_text="Precio real (millones COP)", row=2, col=1)
tablero.update_yaxes(title_text="Cambio en el precio (millones COP)",
                     row=2, col=2)
tablero.write_html(TABLERO / "dashboard.html", include_plotlyjs="cdn")

# Versiones PNG para el informe escrito: el HTML es el entregable navegable,
# pero el documento necesita una imagen fija.
for figura, nombre in [(dispersion, "dispersion_interactiva"),
                       (tablero, "dashboard")]:
    try:
        figura.write_image(TABLERO / f"{nombre}.png", width=1200,
                           height=520 if nombre == "dispersion_interactiva" else 860,
                           scale=2)
    except Exception as error:  # kaleido necesita un navegador instalado
        print(f"Aviso: no se pudo exportar {nombre}.png ({error}). "
              "El HTML interactivo sí quedó generado.")

print("OK - Plotly: dashboard.html y dispersion_interactiva.html en "
      "figures/python/dashboard")
print("\nOK - Fase 4: 3 figuras de seaborn y 2 piezas interactivas de Plotly")
