"""Construcción de la aplicación Dash: layout más callbacks.

Se separa de ``app.py`` para que el objeto ``app`` pueda importarse desde un
servidor WSGI (gunicorn, waitress) sin ejecutar el bloque de arranque.
"""

from pathlib import Path

import dash

from src.callbacks import register_callbacks
from src.layout import layout

BASE_DIR = Path(__file__).resolve().parents[1]

# Dash busca la carpeta de recursos estáticos junto al módulo donde se crea la
# aplicación, que aquí es src/. Como la hoja de estilos vive en la raíz del
# proyecto -junto a app.py, que es donde se espera encontrarla-, hay que
# indicarle la ruta de forma explícita; sin esto el CSS no se sirve y los
# controles se quedan con la apariencia por defecto de Dash.
app = dash.Dash(
    __name__,
    title="Análisis de regresión - Precio de vivienda",
    update_title="Reestimando...",
    assets_folder=str(BASE_DIR / "assets"),
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = layout
register_callbacks(app)
