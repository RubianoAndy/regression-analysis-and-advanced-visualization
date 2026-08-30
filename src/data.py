"""Carga de datos y motor de regresión del dashboard.

Este módulo hace dos cosas.

**Primera: recalcula, no filtra.** Cuando el usuario restringe la vista a un
estrato o a un rango de área, el dashboard no se limita a esconder puntos del
gráfico: vuelve a ajustar la regresión por mínimos cuadrados sobre el
subconjunto elegido. Es una diferencia de fondo. El modelo de los apartamentos
de estrato 5 no es el modelo global mirado de cerca: la pendiente del metro
cuadrado dentro de ese grupo es otra, y sus intervalos de confianza también.
Filtrar y reestimar responden a preguntas distintas, y la que interesa aquí es
la segunda.

**Segunda: se protege de los subconjuntos degenerados.** Al filtrar por un solo
estrato esa columna se vuelve constante y la matriz de diseño pierde rango, de
modo que el ajuste fallaría. La función ``ajustar`` descarta los regresores sin
variación y lo informa, en lugar de dejar que el error llegue a la interfaz.

El coste de reestimar con 150 apartamentos es de milisegundos, así que no hace
falta memoria intermedia.
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DIR = BASE_DIR / "public"
DATA_DIR = BASE_DIR / "data"

# Mismas variables y misma respuesta que utils/codes/python/regression.py.
OBJETIVO = "precio_millones_cop"
PREDICTORAS = ["area_m2", "habitaciones", "antiguedad_anios", "estrato"]

ETIQUETAS = {
    "area_m2": "Área (por m²)",
    "habitaciones": "Habitaciones (por unidad)",
    "antiguedad_anios": "Antigüedad (por año)",
    "estrato": "Estrato (por nivel)",
}

# Mínimo de apartamentos para que el ajuste sea interpretable. Con menos de
# cinco observaciones por parámetro los errores estándar se disparan y los
# intervalos de confianza dejan de significar nada.
MINIMO_APARTAMENTOS = 20


def encode_image(path):
    """Codifica una imagen como URI de datos para incrustarla en el HTML.

    Las imágenes viven en ``public/``, que Dash no sirve de forma automática
    (solo sirve ``assets/``); incrustarlas evita duplicar la carpeta.
    """
    path = Path(path)
    if not path.exists():
        return ""
    datos = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{datos}"


IMAGENES_DIR = PUBLIC_DIR / "assets" / "images"
LOGO_SRC = encode_image(IMAGENES_DIR / "UnisalleDarkLogoV1.png")
AUTHOR_SRC = encode_image(IMAGENES_DIR / "author" / "Andy Rubiano.png")

# --- Datos -------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "dataset" / "viviendas.csv")

ESTRATOS = sorted(df["estrato"].unique())
estrato_options = [{"label": f"Estrato {e}", "value": int(e)} for e in ESTRATOS]

AREA_MIN = float(np.floor(df["area_m2"].min()))
AREA_MAX = float(np.ceil(df["area_m2"].max()))
ANTIGUEDAD_MAX = int(df["antiguedad_anios"].max())
PRECIO_MEDIO_GLOBAL = float(df[OBJETIVO].mean())


def filtrar(estratos, rango_area, antiguedad_max):
    """Devuelve el subconjunto de apartamentos que cumple los tres filtros."""
    sub = df
    if estratos:
        sub = sub[sub["estrato"].isin(estratos)]
    if rango_area:
        sub = sub[sub["area_m2"].between(rango_area[0], rango_area[1])]
    if antiguedad_max is not None:
        sub = sub[sub["antiguedad_anios"] <= antiguedad_max]
    return sub


def ajustar(sub, especificacion):
    """Reestima la regresión sobre ``sub`` y devuelve el modelo y sus métricas.

    ``especificacion`` vale ``'simple'`` (solo el área) o ``'multiple'`` (las
    cuatro características). Devuelve ``None`` si la muestra es insuficiente o
    si no queda ningún regresor con variación.

    Los regresores constantes dentro del subconjunto se descartan: al filtrar
    por un solo estrato, por ejemplo, esa columna deja de aportar información y
    haría singular la matriz de diseño.
    """
    if len(sub) < MINIMO_APARTAMENTOS:
        return None

    candidatas = ["area_m2"] if especificacion == "simple" else PREDICTORAS
    usadas = [v for v in candidatas if sub[v].nunique() > 1]
    descartadas = [v for v in candidatas if v not in usadas]
    if not usadas:
        return None

    modelo = smf.ols(f"{OBJETIVO} ~ " + " + ".join(usadas), data=sub).fit()

    residuos = modelo.resid
    intervalos = modelo.conf_int()
    coeficientes = pd.DataFrame({
        "termino": [ETIQUETAS.get(t, t) for t in modelo.params.index],
        "variable": list(modelo.params.index),
        "coeficiente": modelo.params.values,
        "error_estandar": modelo.bse.values,
        "estadistico_t": modelo.tvalues.values,
        "p_valor": modelo.pvalues.values,
        "ic_inferior": intervalos[0].values,
        "ic_superior": intervalos[1].values,
    })

    return {
        "modelo": modelo,
        "usadas": usadas,
        "descartadas": descartadas,
        "coeficientes": coeficientes,
        "n": int(len(sub)),
        "r2": float(modelo.rsquared),
        "r2_ajustado": float(modelo.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(residuos ** 2))),
        "mae": float(np.mean(np.abs(residuos))),
        "precio_medio": float(sub[OBJETIVO].mean()),
        "ajustados": modelo.fittedvalues,
        "residuos": residuos,
    }


def banda_ajuste(resultado, sub, puntos=120):
    """Malla de predicción sobre el área para dibujar la recta y su banda.

    Las demás variables se fijan en su valor medio dentro del subconjunto, de
    modo que la curva representa el precio esperado de un apartamento
    *promedio* en todo lo que no sea el área.
    """
    modelo = resultado["modelo"]
    malla = np.linspace(sub["area_m2"].min(), sub["area_m2"].max(), puntos)
    escenario = pd.DataFrame({"area_m2": malla})
    for v in resultado["usadas"]:
        if v != "area_m2":
            escenario[v] = sub[v].mean()
    prediccion = modelo.get_prediction(escenario).summary_frame(alpha=0.05)
    return malla, prediccion
