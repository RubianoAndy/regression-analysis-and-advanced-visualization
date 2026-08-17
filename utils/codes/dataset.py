"""Actividad 4 - Fase 0: generación del conjunto de datos.

Regenera con la semilla fija 42 el mismo conjunto de 120 clientes que vienen
usando las actividades anteriores, de modo que todo el esfuerzo de esta
actividad se concentre en modelar los datos y no en describirlos.

Rutas: el script se ubica en codes -> utils -> raíz del proyecto y escribe el
CSV en ``data/dataset``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

"""Tres sectores con niveles de consumo y tarifas distintas: es justamente esa
heterogeneidad la que el modelo simple ignora y el modelo múltiple recupera.
El ruido multiplicativo del 4 % evita que la relación sea exacta."""
rng = np.random.default_rng(42)
n = 120
sectores = rng.choice(
    ["Residencial", "Comercial", "Industrial"], size=n, p=[0.5, 0.3, 0.2]
)
base = {"Residencial": 250, "Comercial": 900, "Industrial": 2500}
amplitud = {"Residencial": 60, "Comercial": 220, "Industrial": 600}
tarifa = {"Residencial": 820, "Comercial": 710, "Industrial": 640}

consumo = np.array([rng.normal(base[s], amplitud[s]) for s in sectores]).clip(50)
costo = consumo * np.array([tarifa[s] for s in sectores]) * rng.normal(1, 0.04, n) / 1000

df = pd.DataFrame({
    "cliente_id": [f"CL-{i:03d}" for i in range(1, n + 1)],
    "sector": sectores,
    "consumo_kwh": consumo.round(1),
    "costo_miles_cop": costo.round(1),
})
df.to_csv(DATA_DIR / "consumo_energia.csv", index=False)

print(f"Dataset generado: {len(df)} clientes")
print(df.groupby("sector")[["consumo_kwh", "costo_miles_cop"]].mean().round(1))
print("\nOK - Fase 0: data/dataset/consumo_energia.csv")
