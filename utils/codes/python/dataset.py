"""Fase 1 - Generación del conjunto de datos.

Crea un conjunto reproducible de 150 apartamentos usados en Bogotá con las
cuatro características que un avalúo mira primero: área, número de
habitaciones, antigüedad y estrato. El precio se construye a partir de ellas
más un ruido aleatorio, así que existe una relación real que la regresión
debe recuperar.

Ejecutar desde la raíz del proyecto:
    python utils/codes/python/dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SALIDA = PROJECT_ROOT / "data" / "dataset"
SALIDA.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 150

# Cada variable se sortea de forma independiente para que los coeficientes de
# la regresión múltiple se puedan leer sin ambigüedad: el efecto de una
# variable no queda contaminado por el de otra.
area = rng.uniform(45, 140, n).round(1)
habitaciones = rng.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.35, 0.15])
antiguedad = rng.integers(0, 36, n)
estrato = rng.choice([3, 4, 5], n, p=[0.40, 0.35, 0.25])

# Precio "verdadero" en millones de COP. Estos son los valores que la
# regresión múltiple tendrá que estimar a partir de los datos.
precio = (
    20                      # valor base
    + 3.8 * area            # millones por metro cuadrado
    + 15 * habitaciones     # millones por habitación adicional
    - 2.5 * antiguedad      # depreciación anual
    + 90 * (estrato - 3)    # sobreprecio por cada nivel de estrato
    + rng.normal(0, 40, n)  # todo lo que las cuatro variables no explican
)

df = pd.DataFrame({
    "inmueble_id": [f"AP-{i:03d}" for i in range(1, n + 1)],
    "area_m2": area,
    "habitaciones": habitaciones,
    "antiguedad_anios": antiguedad,
    "estrato": estrato,
    "precio_millones_cop": precio.round(1),
})
df.to_csv(SALIDA / "viviendas.csv", index=False)

print(f"Dataset generado: {len(df)} apartamentos")
print(df.describe().round(1).to_string())
print("\nOK - Fase 1: data/dataset/viviendas.csv")
