"""Punto de entrada del dashboard interactivo.

Levanta el servidor de desarrollo de Dash en http://localhost:8050/

    python app.py

Los datos deben existir antes de arrancar: el dashboard lee
``data/dataset/viviendas.csv``, que produce la fase 1 del pipeline de
``utils/codes/python``.
"""

from src.dashboard import app, server

__all__ = ["app", "server"]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
