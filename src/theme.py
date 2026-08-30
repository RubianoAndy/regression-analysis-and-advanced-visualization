"""Paleta, tipografía y componentes reutilizables del dashboard.

Los colores institucionales son los de la Universidad de La Salle. Los colores
por estrato son **los mismos** que usan las figuras estáticas de
``utils/codes/python``, de modo que un estrato tiene el mismo color en el
informe impreso y en el navegador.

Aquí viven también los dos componentes que se repiten en toda la interfaz -la
tarjeta de indicador y el contenedor de tarjeta-, para que el módulo de layout
describa la estructura de la página y no sus estilos.
"""

from dash import html

# --- Barra lateral -----------------------------------------------------------
SIDEBAR_BG = "#002D57"
SIDEBAR_TEXT = "#FFFFFF"
SIDEBAR_MUTED = "#7A99B8"

# --- Lienzo principal --------------------------------------------------------
MAIN_BG = "#E6E6E6"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E1E5EE"
SHADOW = "0 2px 12px rgba(0,0,0,0.07)"

TITLE_COLOR = "#002D57"
TEXT_COLOR = "#3D4A5C"
TEXT_MUTED = "#94A3B8"

GOLD = "#FFCD00"
NARANJA = "#d95f02"
GRIS = "#9e9e9e"

# Mismos colores que COLORES_ESTRATO en visualization.py.
COLORES_ESTRATO = {3: "#a6bddb", 4: "#4292c6", 5: "#08519c"}

FONT = '"Segoe UI", Arial, sans-serif'

BASE_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family=FONT, size=12),
    margin=dict(l=62, r=28, t=58, b=48),
)

dd_container_style = {"marginBottom": "18px"}


def color_estrato(e):
    """Color asignado a un estrato."""
    return COLORES_ESTRATO.get(int(e), TITLE_COLOR)


def title_cfg(text, subtitle=None):
    """Configuración de título para las figuras de Plotly del dashboard.

    Plotly no tiene subtítulo nativo: se incrusta como segunda línea del título
    con su propio tamaño y color.
    """
    html_text = f"<b>{text}</b>"
    if subtitle:
        html_text += (
            f"<br><span style='font-size:11px;color:{TEXT_MUTED}'>{subtitle}</span>"
        )
    return dict(text=html_text, font=dict(size=14, color=TITLE_COLOR),
                x=0.012, xanchor="left", y=0.97, yanchor="top")


def kpi_card(icon, value, label, card_id=None, hint=None):
    """Tarjeta de indicador: icono, valor grande y etiqueta en versalitas."""
    return html.Div(
        [
            html.Div(icon, style={"fontSize": "24px", "marginBottom": "8px"}),
            html.Div(value, id=card_id, style={
                "fontSize": "25px", "fontWeight": "700",
                "color": TITLE_COLOR, "lineHeight": "1",
            }),
            html.Div(label, style={
                "fontSize": "9.5px", "color": TEXT_MUTED, "marginTop": "6px",
                "textTransform": "uppercase", "letterSpacing": "0.8px",
                "fontWeight": "600",
            }),
        ],
        title=hint or "",
        style={
            "backgroundColor": CARD_BG,
            "border": f"1px solid {CARD_BORDER}",
            "padding": "18px 22px",
            "flex": "1",
            "boxShadow": SHADOW,
            "minWidth": "150px",
            "fontFamily": FONT,
        },
    )


def card(children, flex=1, padding=False, min_w="0"):
    """Contenedor blanco con borde y sombra: la unidad visual del lienzo."""
    style = {
        "flex": str(flex),
        "minWidth": min_w,
        "backgroundColor": CARD_BG,
        "border": f"1px solid {CARD_BORDER}",
        "boxShadow": SHADOW,
        "overflow": "hidden",
    }
    if padding:
        style["padding"] = "18px 20px"
    return html.Div(children, style=style)


def section_label(text):
    """Rótulo de sección en la barra lateral."""
    return html.Div(text, style={
        "color": SIDEBAR_MUTED, "fontSize": "10px", "fontWeight": "700",
        "letterSpacing": "1.4px", "marginBottom": "14px",
    })


def field_label(text):
    """Etiqueta de un control de la barra lateral."""
    return html.Label(text, style={
        "color": SIDEBAR_TEXT, "fontSize": "11px", "fontWeight": "600",
        "display": "block", "marginBottom": "6px", "letterSpacing": "0.3px",
    })
