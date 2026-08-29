"""Fase 3 - Validación de los modelos con scikit-learn.

statsmodels dice qué tan bien se ajusta un modelo a los datos que ya vio.
scikit-learn responde la pregunta que de verdad importa: ¿acierta con
apartamentos que nunca vio?

Se hacen dos comprobaciones sobre las mismas dos especificaciones de la Fase 2:

* partición 70 / 30 en entrenamiento y prueba;
* validación cruzada de 5 pliegues sobre todo el conjunto.

Ejecutar desde la raíz del proyecto:
    python utils/codes/python/validation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

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
AZUL, NARANJA, CLARO = "#2b8cbe", "#d95f02", "#a6bddb"
SEMILLA = 42

df = pd.read_csv(DATASET)
y = df["precio_millones_cop"]

# Las dos especificaciones que se comparan: solo el área frente a las cuatro
# variables. Son exactamente las de la Fase 2, reescritas para scikit-learn.
ESPECIFICACIONES = {
    "Simple": ["area_m2"],
    "Múltiple": ["area_m2", "habitaciones", "antiguedad_anios", "estrato"],
}

# 1. Partición entrenamiento / prueba.
# El 30 % de los apartamentos se aparta antes de ajustar nada: el modelo no los
# ve durante el entrenamiento y sirven para medir el error real de predicción.
indices = np.arange(len(df))
idx_train, idx_test = train_test_split(indices, test_size=0.30,
                                       random_state=SEMILLA)
print(f"Entrenamiento: {len(idx_train)} apartamentos | "
      f"Prueba: {len(idx_test)} apartamentos")

# 2. Ajuste, evaluación y validación cruzada de cada especificación.
# Un solo corte 70/30 depende del azar de qué 45 apartamentos tocaron la
# prueba. La validación cruzada repite el experimento 5 veces con particiones
# distintas, así que su promedio es una estimación más estable.
cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
filas, predicciones, r2_pliegues = [], {}, {}

for nombre, columnas in ESPECIFICACIONES.items():
    X = df[columnas]
    modelo = LinearRegression().fit(X.iloc[idx_train], y.iloc[idx_train])
    y_pred = modelo.predict(X.iloc[idx_test])
    predicciones[nombre] = y_pred

    puntajes = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    r2_pliegues[nombre] = puntajes

    filas.append({
        "modelo": nombre,
        "variables": len(columnas),
        "r2_entrenamiento": round(
            r2_score(y.iloc[idx_train], modelo.predict(X.iloc[idx_train])), 4),
        "r2_prueba": round(r2_score(y.iloc[idx_test], y_pred), 4),
        "rmse_prueba": round(
            float(np.sqrt(mean_squared_error(y.iloc[idx_test], y_pred))), 2),
        "mae_prueba": round(mean_absolute_error(y.iloc[idx_test], y_pred), 2),
        "r2_cv_media": round(float(puntajes.mean()), 4),
        "r2_cv_desviacion": round(float(puntajes.std()), 4),
    })

metricas = pd.DataFrame(filas)
metricas.to_csv(TABLAS / "validacion_sklearn.csv", index=False)
print("\nDesempeño sobre datos que el modelo no vio")
print(metricas.to_string(index=False))

mejor = metricas.loc[metricas["r2_prueba"].idxmax(), "modelo"]
error_medio = metricas.loc[metricas["modelo"] == mejor, "mae_prueba"].iloc[0]
print(f"\nMejor modelo fuera de la muestra: {mejor}. "
      f"Se equivoca en promedio {error_medio:.1f} millones de COP por "
      f"apartamento, sobre un precio medio de {y.mean():.1f} millones "
      f"({error_medio / y.mean():.1%}).")

# El R² de entrenamiento y el de prueba son casi iguales: no hay
# sobreajuste, que es el riesgo típico al añadir variables a un modelo.
brecha = (metricas.loc[metricas["modelo"] == mejor, "r2_entrenamiento"].iloc[0]
          - metricas.loc[metricas["modelo"] == mejor, "r2_prueba"].iloc[0])
print(f"Brecha entrenamiento - prueba: {brecha:+.4f} "
      f"({'sin sobreajuste' if abs(brecha) < 0.05 else 'revisar sobreajuste'}).")

# 3. Figura de validación.
# Izquierda: cada punto es un apartamento de prueba; la diagonal es la
# predicción perfecta y la distancia a ella es el error cometido.
# Derecha: el R² pliegue a pliegue, para ver que la ventaja del modelo
# múltiple se sostiene en las cinco particiones y no fue suerte de una.
y_test = y.iloc[idx_test]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))

for nombre, color in [("Simple", CLARO), ("Múltiple", AZUL)]:
    ax1.scatter(predicciones[nombre], y_test, s=34, color=color,
                edgecolor="white", linewidth=0.5, label=f"Modelo {nombre.lower()}")
tope = [float(y_test.min()) * 0.85, float(y_test.max()) * 1.08]
ax1.plot(tope, tope, color=NARANJA, lw=1.5, linestyle="--",
         label="Predicción perfecta")
ax1.set_title("Conjunto de prueba: precio real frente al predicho")
ax1.set_xlabel("Precio predicho (millones de COP)")
ax1.set_ylabel("Precio real (millones de COP)")
ax1.legend(fontsize=8, loc="upper left")

ancho = 0.38
posicion = np.arange(1, cv.get_n_splits() + 1)
for desplazamiento, (nombre, color) in zip(
        (-ancho / 2, ancho / 2), [("Simple", CLARO), ("Múltiple", AZUL)]):
    barras = ax2.bar(posicion + desplazamiento, r2_pliegues[nombre], ancho,
                     color=color, label=f"Modelo {nombre.lower()}")
    ax2.bar_label(barras, fmt="%.2f", fontsize=7, padding=2)
ax2.set_ylim(0, 1.05)
ax2.set_xticks(posicion, [f"Pliegue {i}" for i in posicion], fontsize=8)
ax2.set_title("Validación cruzada de 5 pliegues: R² en cada partición")
ax2.set_ylabel("R² sobre el pliegue de prueba")
ax2.legend(fontsize=8, loc="lower right")

fig.suptitle("La ventaja del modelo múltiple se mantiene fuera de la muestra",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURAS / "validacion_sklearn.png")
plt.close(fig)

print("\nOK - Fase 3: validacion_sklearn.csv y 1 figura en "
      "figures/python/regression")
