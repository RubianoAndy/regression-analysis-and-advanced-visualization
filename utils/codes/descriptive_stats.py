"""Actividad 2 - Análisis estadístico de datos y creación de gráficos básicos.

Genera (o regenera con la misma semilla de la Actividad 1) el dataset de
consumo energético, construye la distribución de frecuencias por el método
de Sturges, calcula las medidas de tendencia central (media, mediana, moda)
y de dispersión (rango, varianza, desviación estándar, coeficiente de
variación) y produce los gráficos básicos con Matplotlib.

Rutas: el script se ubica en codes -> utils -> raíz del proyecto.
Escribe el CSV en ``data/dataset``, las tablas estadísticas en
``data/processed`` y las imágenes en
``public/assets/images/figures/python/statistics/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = (
    PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python" / "statistics"
)
for d in (DATA_DIR, PROCESSED_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

"""Dataset: idéntico al de la Actividad 1 (semilla fija 42) para dar
continuidad al análisis sobre las mismas 120 observaciones."""
rng = np.random.default_rng(42)
n = 120
sectors = rng.choice(
    ["Residencial", "Comercial", "Industrial"], size=n, p=[0.5, 0.3, 0.2]
)
base = {"Residencial": 250, "Comercial": 900, "Industrial": 2500}
spread = {"Residencial": 60, "Comercial": 220, "Industrial": 600}
consumption = np.array([rng.normal(base[s], spread[s]) for s in sectors]).clip(50)
tariff = {"Residencial": 820, "Comercial": 710, "Industrial": 640}
cost = consumption * np.array([tariff[s] for s in sectors]) * rng.normal(1, 0.04, n) / 1000

df = pd.DataFrame({
    "cliente_id": [f"CL-{i:03d}" for i in range(1, n + 1)],
    "sector": sectors,
    "consumo_kwh": consumption.round(1),
    "costo_miles_cop": cost.round(1),
})
df.to_csv(DATA_DIR / "consumo_energia.csv", index=False)

"""1. DISTRIBUCIÓN DE FRECUENCIAS (regla de Sturges).

k = 1 + 3.322*log10(n) clases de igual amplitud sobre el consumo global.
La tabla incluye límites, marca de clase, frecuencias absolutas (fi),
acumuladas (Fi), relativas (hi %) y relativas acumuladas (Hi %).
"""
x = df["consumo_kwh"]
k = int(np.ceil(1 + 3.322 * np.log10(n)))
edges = np.linspace(x.min(), x.max(), k + 1)
fi, _ = np.histogram(x, bins=edges)
freq_table = pd.DataFrame({
    "lim_inferior": edges[:-1].round(1),
    "lim_superior": edges[1:].round(1),
    "marca_clase": ((edges[:-1] + edges[1:]) / 2).round(1),
    "fi": fi,
    "Fi": np.cumsum(fi),
    "hi_pct": (fi / n * 100).round(1),
    "Hi_pct": (np.cumsum(fi) / n * 100).round(1),
})
freq_table.to_csv(PROCESSED_DIR / "freq_table.csv", index=False)
print(f"Distribución de frecuencias ({k} clases, amplitud {edges[1]-edges[0]:.1f} kWh)")
print(freq_table.to_string(index=False))

"""2. MEDIDAS DE TENDENCIA CENTRAL.

La moda del consumo (variable continua) se reporta como clase modal e
interpolación: Mo = L + d1/(d1+d2) * w. La moda de la variable nominal
``sector`` es simplemente la categoría más frecuente.
"""


def interpolated_mode(values: pd.Series, bin_edges: np.ndarray) -> float:
    # Moda interpolada dentro de la clase modal (fórmula de agrupados).
    counts, _ = np.histogram(values, bins=bin_edges)
    m = int(np.argmax(counts))
    width = bin_edges[m + 1] - bin_edges[m]
    d1 = counts[m] - (counts[m - 1] if m > 0 else 0)
    d2 = counts[m] - (counts[m + 1] if m < len(counts) - 1 else 0)
    if d1 + d2 == 0:
        return float(bin_edges[m] + width / 2)
    return float(bin_edges[m] + d1 / (d1 + d2) * width)


central_rows = []
for label, g in [("Residencial", None), ("Comercial", None),
                 ("Industrial", None), ("Global", None)]:
    subset = x if label == "Global" else df.loc[df["sector"] == label, "consumo_kwh"]
    k_g = int(np.ceil(1 + 3.322 * np.log10(len(subset))))
    edges_g = np.linspace(subset.min(), subset.max(), k_g + 1)
    central_rows.append({
        "grupo": label,
        "n": len(subset),
        "media": round(subset.mean(), 1),
        "mediana": round(subset.median(), 1),
        "moda_interpolada": round(interpolated_mode(subset, edges_g), 1),
    })
central = pd.DataFrame(central_rows)
central.to_csv(PROCESSED_DIR / "central_tendency.csv", index=False)
print("\nMedidas de tendencia central del consumo (kWh)")
print(central.to_string(index=False))
print(f"Moda del sector (variable nominal): {df['sector'].mode()[0]}")

"""3. MEDIDAS DE DISPERSIÓN.

Varianza y desviación estándar muestrales (ddof=1), rango, coeficiente de
variación y rango intercuartílico, por sector y a nivel global.
"""
dispersion_rows = []
for label in ["Residencial", "Comercial", "Industrial", "Global"]:
    subset = x if label == "Global" else df.loc[df["sector"] == label, "consumo_kwh"]
    q1, q3 = subset.quantile([0.25, 0.75])
    dispersion_rows.append({
        "grupo": label,
        "rango": round(subset.max() - subset.min(), 1),
        "varianza": round(subset.var(ddof=1), 1),
        "desv_std": round(subset.std(ddof=1), 1),
        "cv_pct": round(subset.std(ddof=1) / subset.mean() * 100, 1),
        "q1": round(q1, 1),
        "q3": round(q3, 1),
        "iqr": round(q3 - q1, 1),
    })
dispersion = pd.DataFrame(dispersion_rows)
dispersion.to_csv(PROCESSED_DIR / "dispersion.csv", index=False)
print("\nMedidas de dispersión del consumo (kWh)")
print(dispersion.to_string(index=False))

"""4. GRÁFICOS BÁSICOS.

Principios aplicados (Actividad 1): título informativo, ejes con unidades,
eje de frecuencias desde cero, cuadrícula sutil detrás de los datos, color
funcional y etiquetas de datos.
"""

"""Histograma con las clases de Sturges y las tres medidas de tendencia
central superpuestas: evidencia la asimetría positiva del consumo global."""
global_mean = x.mean()
global_median = x.median()
global_mode = interpolated_mode(x, edges)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(x, bins=edges, color="#2c7fb8", edgecolor="white")
for value, name, style in [
    (global_mode, f"Moda = {global_mode:,.0f}", ":"),
    (global_median, f"Mediana = {global_median:,.0f}", "--"),
    (global_mean, f"Media = {global_mean:,.0f}", "-"),
]:
    ax.axvline(value, color="#d95f02", linestyle=style, lw=1.4, label=name)
ax.set_title(f"Distribución de frecuencias del consumo ({k} clases de Sturges)")
ax.set_xlabel("Consumo (kWh/mes)")
ax.set_ylabel("Frecuencia absoluta (clientes)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "hist_sturges_central_tendency.png")
plt.close(fig)

"""Polígono de frecuencias y ojiva: dos vistas complementarias de la misma
tabla, la puntual (marcas de clase vs fi) y la acumulada (límites
superiores vs Hi %)."""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
poly_x = np.concatenate(([edges[0]], freq_table["marca_clase"], [edges[-1]]))
poly_y = np.concatenate(([0], freq_table["fi"], [0]))
ax1.plot(poly_x, poly_y, marker="o", color="#2b8cbe")
ax1.set_title("Polígono de frecuencias")
ax1.set_xlabel("Consumo (kWh/mes)")
ax1.set_ylabel("Frecuencia absoluta (clientes)")
ax2.plot(np.concatenate(([edges[0]], freq_table["lim_superior"])),
         np.concatenate(([0], freq_table["Hi_pct"])),
         marker="o", color="#2b8cbe")
ax2.axhline(50, color="#d95f02", linestyle="--", lw=1,
            label=f"50 % -> Mediana ≈ {global_median:,.0f} kWh")
ax2.set_title("Ojiva (frecuencia acumulada)")
ax2.set_xlabel("Consumo (kWh/mes)")
ax2.set_ylabel("Frecuencia acumulada (%)")
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "freq_polygon_ogive.png")
plt.close(fig)

"""Barras de frecuencia absoluta por sector: distribución de la variable
nominal, ordenada por magnitud y con etiquetas n (%)."""
sector_freq = df["sector"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(sector_freq.index, sector_freq.values,
              color=["#a6bddb", "#74a9cf", "#2b8cbe"])
ax.bar_label(bars, labels=[f"{v} ({v/n:.0%})" for v in sector_freq.values], padding=3)
ax.set_ylim(0, sector_freq.max() * 1.18)
ax.set_title("Frecuencia de clientes por sector (moda nominal: "
             f"{df['sector'].mode()[0]})")
ax.set_xlabel("Sector")
ax.set_ylabel("Frecuencia absoluta (clientes)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "bar_freq_by_sector.png")
plt.close(fig)

"""Barras agrupadas media vs mediana por sector: la cercanía de ambas
dentro de cada grupo evidencia simetría local pese a la asimetría global."""
sector_order = ["Residencial", "Comercial", "Industrial"]
central_plot = central.set_index("grupo").loc[sector_order]
pos = np.arange(len(sector_order))
width = 0.38
fig, ax = plt.subplots(figsize=(6.5, 3.6))
b1 = ax.bar(pos - width / 2, central_plot["media"], width,
            label="Media", color="#2b8cbe")
b2 = ax.bar(pos + width / 2, central_plot["mediana"], width,
            label="Mediana", color="#a6bddb")
ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
ax.set_xticks(pos, sector_order)
ax.set_ylim(0, central_plot["media"].max() * 1.18)
ax.set_title("Media y mediana del consumo por sector")
ax.set_xlabel("Sector")
ax.set_ylabel("Consumo (kWh/mes)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "bar_mean_median_by_sector.png")
plt.close(fig)

"""Diagrama de caja con la media superpuesta: lectura conjunta de posición
(mediana), dispersión (IQR y bigotes) y su relación con sigma."""
data = [df.loc[df["sector"] == s, "consumo_kwh"] for s in sector_order]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
bp = ax.boxplot(data, tick_labels=sector_order, patch_artist=True,
                medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], ["#a6bddb", "#74a9cf", "#2b8cbe"]):
    patch.set_facecolor(c)
sigma = dispersion.set_index("grupo").loc[sector_order, "desv_std"]
means = central.set_index("grupo").loc[sector_order, "media"]
ax.scatter(pos + 1, means, color="#d95f02", zorder=3, s=30, label="Media")
for i, s in enumerate(sector_order):
    ax.annotate(f"σ = {sigma[s]:,.0f}", (i + 1, means[s]),
                xytext=(8, -4), textcoords="offset points", fontsize=8)
ax.set_title("Dispersión del consumo por sector (mediana, IQR, media y σ)")
ax.set_xlabel("Sector")
ax.set_ylabel("Consumo (kWh/mes)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "boxplot_dispersion_by_sector.png")
plt.close(fig)

print("\nOK - tablas estadísticas y figuras de Python generadas")
