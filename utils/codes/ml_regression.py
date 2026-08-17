"""Actividad 4 - Fase 3: los mismos modelos con scikit-learn y validación.

statsmodels responde si un coeficiente es significativo; scikit-learn responde
si el modelo predice bien datos que nunca vio. Esta fase repite las
especificaciones de la Fase 2 dentro de un ``Pipeline`` y las somete a
partición entrenamiento/prueba, validación cruzada de 10 pliegues, curva de
aprendizaje e importancia por permutación.

Se comparan cinco estimadores sobre la misma matriz de diseño con interacción:
mínimos cuadrados sin penalizar, las tres regularizaciones clásicas (Ridge,
Lasso y ElasticNet) y una versión que aprende sobre el logaritmo del costo
mediante ``TransformedTargetRegressor``, equivalente al modelo M4.

Rutas: el script se ubica en codes -> utils -> raíz del proyecto.
Lee el CSV de ``data/dataset``, escribe las tablas en ``data/processed`` y
las imágenes en ``public/assets/images/figures/python/regression/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

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

SEMILLA = 42
SECTOR_ORDER = ["Residencial", "Comercial", "Industrial"]
SECTOR_COLORS = {"Residencial": "#a6bddb", "Comercial": "#74a9cf",
                 "Industrial": "#2b8cbe"}
ACCENT = "#d95f02"

df = pd.read_csv(DATA_DIR / "consumo_energia.csv")
X = df[["consumo_kwh", "sector"]]
y = df["costo_miles_cop"]

"""1. PARTICIÓN ESTRATIFICADA.

La estratificación por sector es obligatoria aquí: el sector Industrial tiene
solo 18 clientes, y una partición aleatoria simple podría dejar el conjunto de
prueba casi sin representación del grupo que más factura.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEMILLA, stratify=df["sector"])
print(f"Entrenamiento: {len(X_train)} clientes | Prueba: {len(X_test)} clientes")
print(pd.concat([
    X_train["sector"].value_counts().rename("entrenamiento"),
    X_test["sector"].value_counts().rename("prueba"),
], axis=1).loc[SECTOR_ORDER].to_string())

"""2. MATRIZ DE DISEÑO COMÚN.

El preprocesamiento codifica el sector en variables indicadoras (descartando
la primera para evitar la trampa de la variable ficticia), estandariza el
consumo y genera los productos consumo x sector. Encapsularlo en un
``Pipeline`` garantiza que el ajuste del escalador se calcule solo con los
datos de entrenamiento, sin filtrar información del conjunto de prueba.
"""


def construir_pipeline(estimador):
    preprocesamiento = ColumnTransformer([
        ("numerica", StandardScaler(), ["consumo_kwh"]),
        ("categorica", OneHotEncoder(drop="first", sparse_output=False),
         ["sector"]),
    ])
    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("interaccion", PolynomialFeatures(degree=2, interaction_only=True,
                                           include_bias=False)),
        ("estimador", estimador),
    ])


"""Los tres estimadores penalizados dependen de un hiperparámetro que no puede
fijarse a ojo: se busca en una rejilla logarítmica con validación cruzada
interna, de modo que la comparación con MCO sea justa y no un empate amañado
por una α mal elegida.
"""
REJILLA = {"estimador__alpha": np.logspace(-3, 3, 13)}


def con_busqueda(estimador, rejilla=REJILLA):
    return GridSearchCV(construir_pipeline(estimador), rejilla,
                        scoring="neg_root_mean_squared_error",
                        cv=KFold(n_splits=5, shuffle=True,
                                 random_state=SEMILLA))


modelos = {
    "MCO (sin penalización)": construir_pipeline(LinearRegression()),
    "Ridge (L2, α ajustada)": con_busqueda(Ridge()),
    "Lasso (L1, α ajustada)": con_busqueda(Lasso(max_iter=100000)),
    "ElasticNet (α ajustada, l1 = 0,5)": con_busqueda(
        ElasticNet(l1_ratio=0.5, max_iter=100000)),
    "MCO sobre log(costo)": construir_pipeline(
        TransformedTargetRegressor(regressor=LinearRegression(),
                                   func=np.log, inverse_func=np.exp)),
}

"""3. EVALUACIÓN: PRUEBA RETENIDA Y VALIDACIÓN CRUZADA.

La partición única entrega una cifra fácil de comunicar, pero depende de qué
30 clientes tocó el azar. La validación cruzada de 10 pliegues repite el
experimento diez veces y su desviación estándar mide justamente esa
inestabilidad, así que ambas se reportan juntas.
"""
cv = KFold(n_splits=10, shuffle=True, random_state=SEMILLA)
filas, predicciones = [], {}
for nombre, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    pred_test = pipe.predict(X_test)
    predicciones[nombre] = pred_test
    marcador = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["r2", "neg_root_mean_squared_error"])
    alfa = (pipe.best_params_["estimador__alpha"]
            if isinstance(pipe, GridSearchCV) else np.nan)
    filas.append({
        "modelo": nombre,
        "alpha_seleccionada": round(alfa, 4) if alfa == alfa else "-",
        "r2_entrenamiento": round(r2_score(y_train, pipe.predict(X_train)), 4),
        "r2_prueba": round(r2_score(y_test, pred_test), 4),
        "rmse_prueba": round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 2),
        "mae_prueba": round(mean_absolute_error(y_test, pred_test), 2),
        "mape_prueba_pct": round(
            mean_absolute_percentage_error(y_test, pred_test) * 100, 2),
        "r2_cv_media": round(marcador["test_r2"].mean(), 4),
        "r2_cv_desv": round(marcador["test_r2"].std(), 4),
        "rmse_cv_media": round(
            -marcador["test_neg_root_mean_squared_error"].mean(), 2),
        "rmse_cv_desv": round(
            marcador["test_neg_root_mean_squared_error"].std(), 2),
    })
    modelos[nombre] = pipe

metricas = pd.DataFrame(filas)
metricas.to_csv(PROCESSED_DIR / "sklearn_metricas.csv", index=False)
print("\nDesempeño de los cinco estimadores")
print(metricas.to_string(index=False))

"""La selección no se resuelve con el mínimo del RMSE: las diferencias entre
los cuatro primeros estimadores caben dentro del error de la propia
estimación. Se aplica la regla de un error estándar —quedarse con el modelo
más simple cuyo desempeño esté a menos de un error estándar del mejor—, que
convierte un empate estadístico en una decisión reproducible y prefiere la
especificación más interpretable.
"""
idx_min = metricas["rmse_cv_media"].idxmin()
error_estandar = metricas.loc[idx_min, "rmse_cv_desv"] / np.sqrt(cv.get_n_splits())
umbral = metricas.loc[idx_min, "rmse_cv_media"] + error_estandar
candidatos = metricas[metricas["rmse_cv_media"] <= umbral]
mejor = candidatos.iloc[0]["modelo"]
pipe_mejor = modelos[mejor]
print(f"\nMenor RMSE en validación cruzada: {metricas.loc[idx_min, 'modelo']} "
      f"({metricas.loc[idx_min, 'rmse_cv_media']:.2f})")
print(f"Umbral de un error estándar: {umbral:.2f} -> "
      f"{len(candidatos)} modelos empatados")
print(f"Modelo seleccionado por parsimonia: {mejor}")

"""4. IMPORTANCIA POR PERMUTACIÓN.

Permutar una columna destruye su relación con el objetivo; la caída del R2
que provoca mide cuánto dependía el modelo de ella. A diferencia de los
coeficientes, esta medida no se ve afectada por la escala de las variables ni
por la multicolinealidad de la matriz de diseño.
"""
importancia = permutation_importance(
    pipe_mejor, X_test, y_test, n_repeats=30, random_state=SEMILLA,
    scoring="r2")
tabla_importancia = pd.DataFrame({
    "variable": X.columns,
    "caida_media_r2": np.round(importancia.importances_mean, 4),
    "desv_std": np.round(importancia.importances_std, 4),
}).sort_values("caida_media_r2", ascending=False)
tabla_importancia.to_csv(PROCESSED_DIR / "importancia_permutacion.csv",
                         index=False)
print("\nImportancia por permutación sobre el conjunto de prueba")
print(tabla_importancia.to_string(index=False))

"""5. FIGURAS DE VALIDACIÓN."""

"""Observado frente a predicho en el conjunto de prueba: la diagonal es la
predicción perfecta y la distancia vertical a ella es el error de cada
cliente. Los residuos del panel derecho comprueban que el error no crece de
forma sistemática con el tamaño de la factura."""
pred_mejor = predicciones[mejor]
sector_test = df.loc[X_test.index, "sector"]
residuo_test = y_test - pred_mejor

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))
for s in SECTOR_ORDER:
    m = (sector_test == s).to_numpy()
    ax1.scatter(pred_mejor[m], y_test[m], s=34, color=SECTOR_COLORS[s],
                edgecolor="white", linewidth=0.5, label=s)
limite = [0, float(y_test.max()) * 1.08]
ax1.plot(limite, limite, color=ACCENT, lw=1.4, linestyle="--",
         label="Predicción perfecta")
r2_prueba = float(metricas.loc[metricas["modelo"] == mejor, "r2_prueba"].iloc[0])
rmse_prueba = float(metricas.loc[metricas["modelo"] == mejor,
                                 "rmse_prueba"].iloc[0])
ax1.set_title(f"Prueba retenida: R² = {r2_prueba:.4f}, RMSE = {rmse_prueba:,.1f}")
ax1.set_xlabel("Costo predicho (miles de COP)")
ax1.set_ylabel("Costo observado (miles de COP)")
ax1.legend(fontsize=8, loc="upper left")

for s in SECTOR_ORDER:
    m = (sector_test == s).to_numpy()
    ax2.scatter(pred_mejor[m], residuo_test[m], s=34, color=SECTOR_COLORS[s],
                edgecolor="white", linewidth=0.5, label=s)
ax2.axhline(0, color=ACCENT, lw=1.2)
ax2.set_title("Residuos de prueba: sin sesgo visible por sector")
ax2.set_xlabel("Costo predicho (miles de COP)")
ax2.set_ylabel("Residuo (miles de COP)")
ax2.legend(fontsize=8, loc="lower left")
fig.suptitle(f"Validación del modelo seleccionado · {mejor}",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "real_vs_predicho.png")
plt.close(fig)

"""Comparación de los cinco estimadores en los diez pliegues: la caja resume
la distribución del error, no un único número, y su altura advierte cuánto
puede cambiar el resultado según qué clientes toque validar."""
resultados_cv = {}
for nombre, pipe in modelos.items():
    marcador = cross_validate(pipe, X, y, cv=cv,
                              scoring="neg_root_mean_squared_error")
    resultados_cv[nombre] = -marcador["test_score"]

fig, ax = plt.subplots(figsize=(9.6, 4.2))
etiquetas = list(resultados_cv)
bp = ax.boxplot([resultados_cv[k] for k in etiquetas],
                tick_labels=[e.replace(" (", "\n(") for e in etiquetas],
                patch_artist=True, medianprops=dict(color="black"))
for parche, nombre in zip(bp["boxes"], etiquetas):
    parche.set_facecolor("#2b8cbe" if nombre == mejor else "#a6bddb")
for i, nombre in enumerate(etiquetas):
    ax.annotate(f"{resultados_cv[nombre].mean():.1f}",
                (i + 1, resultados_cv[nombre].mean()), xytext=(12, -3),
                textcoords="offset points", fontsize=8, color=ACCENT)
ax.axhline(umbral, color=ACCENT, lw=1.2, linestyle="--",
           label=f"Umbral de un error estándar = {umbral:.1f}")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("RMSE en validación cruzada de 10 pliegues: la penalización no "
             "aporta cuando no sobra información")
ax.set_xlabel("Estimador")
ax.set_ylabel("RMSE (miles de COP)")
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "comparacion_regularizacion.png")
plt.close(fig)

"""Curva de aprendizaje: cuántos clientes necesita el modelo. La convergencia
de ambas curvas indica que el error restante es ruido irreducible y no falta
de datos; una brecha persistente indicaría sobreajuste."""
tamanos, marca_train, marca_val = learning_curve(
    pipe_mejor, X, y, cv=cv, scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.15, 1.0, 10), random_state=SEMILLA)
rmse_train = -marca_train.mean(axis=1)
rmse_val = -marca_val.mean(axis=1)
desv_val = marca_val.std(axis=1)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(tamanos, rmse_train, marker="o", color="#2b8cbe",
        label="Error de entrenamiento")
ax.plot(tamanos, rmse_val, marker="s", color=ACCENT,
        label="Error de validación cruzada")
ax.fill_between(tamanos, rmse_val - desv_val, rmse_val + desv_val,
                color=ACCENT, alpha=0.15)
ax.set_title(f"Curva de aprendizaje · {mejor}")
ax.set_xlabel("Clientes usados para entrenar")
ax.set_ylabel("RMSE (miles de COP)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "curva_aprendizaje.png")
plt.close(fig)

curva = pd.DataFrame({
    "n_entrenamiento": tamanos,
    "rmse_entrenamiento": np.round(rmse_train, 2),
    "rmse_validacion": np.round(rmse_val, 2),
})
curva.to_csv(PROCESSED_DIR / "curva_aprendizaje.csv", index=False)
print("\nCurva de aprendizaje (RMSE en miles de COP)")
print(curva.to_string(index=False))

print("\nOK - Fase 3: validación con scikit-learn y figuras generadas")
