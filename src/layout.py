"""Estructura de la página: barra lateral de controles y lienzo de resultados.

El layout es estático -describe qué componentes existen y dónde-, y todos los
valores llegan desde ``callbacks.py``. Las figuras se declaran vacías y se
rellenan en el primer disparo del callback, que Dash ejecuta al cargar.

La disposición sigue el orden en que se lee un análisis de regresión:
indicadores, luego el ajuste (donde vive el resultado), después el efecto de
cada variable, y al final el diagnóstico -lo observado frente a lo estimado y
los errores- con la tabla de coeficientes.
"""

from dash import dash_table, dcc, html

from src.data import (
    ANTIGUEDAD_MAX, AREA_MAX, AREA_MIN, AUTHOR_SRC, LOGO_SRC,
    PREDICTORAS, df, estrato_options,
)
from src.theme import (
    CARD_BG, CARD_BORDER, FONT, GOLD, MAIN_BG, SIDEBAR_BG, SIDEBAR_MUTED,
    SIDEBAR_TEXT, TEXT_COLOR, TEXT_MUTED, TITLE_COLOR,
    card, dd_container_style, field_label, kpi_card, section_label,
)

SIDEBAR_WIDTH = "250px"

GRAPH_CONFIG = {"displaylogo": False, "displayModeBar": False, "responsive": True}
GRAPH_CONFIG_FULL = {"displaylogo": False, "responsive": True,
                     "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

_marcas_area = {int(v): {"label": f"{int(v)}",
                         "style": {"color": SIDEBAR_MUTED, "fontSize": "10px"}}
                for v in range(int(AREA_MIN), int(AREA_MAX) + 1, 25)}
_marcas_antiguedad = {v: {"label": str(v),
                          "style": {"color": SIDEBAR_MUTED, "fontSize": "10px"}}
                      for v in range(0, ANTIGUEDAD_MAX + 1, 7)}


sidebar = html.Div(
    [
        # --- Logo ------------------------------------------------------------
        html.Div(
            html.A(
                href="https://lasalle.edu.co/",
                target="_blank",
                rel="noopener noreferrer",
                style={"cursor": "pointer", "display": "block"},
                children=html.Img(src=LOGO_SRC, style={
                    "width": "82%", "maxWidth": "168px",
                    "display": "block", "margin": "0 auto",
                }),
            ) if LOGO_SRC else html.Div([
                html.P("UNIVERSIDAD", style={
                    "color": SIDEBAR_TEXT, "fontSize": "11px", "margin": "0",
                    "fontWeight": "700", "letterSpacing": "1px",
                    "textAlign": "center"}),
                html.P("DE LA SALLE", style={
                    "color": GOLD, "fontSize": "13px", "margin": "2px 0 0 0",
                    "fontWeight": "900", "letterSpacing": "1px",
                    "textAlign": "center"}),
            ]),
            style={
                "padding": "20px 16px 18px",
                "borderBottom": "1px solid rgba(255,255,255,0.1)",
                "flexShrink": "0",
            },
        ),

        # --- Controles -------------------------------------------------------
        html.Div(
            [
                section_label("FILTROS"),

                html.Div([
                    field_label("🏙️  Estrato"),
                    dcc.Dropdown(
                        id="filter-estrato",
                        options=estrato_options,
                        value=[],
                        multi=True,
                        placeholder="Todos los estratos",
                        clearable=True,
                        className="sidebar-dropdown",
                    ),
                ], style=dd_container_style),

                html.Div([
                    field_label("📐  Área (m²)"),
                    dcc.RangeSlider(
                        id="filter-area",
                        min=AREA_MIN, max=AREA_MAX, step=5,
                        value=[AREA_MIN, AREA_MAX],
                        marks=_marcas_area,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"marginBottom": "20px"}),

                html.Div([
                    field_label("🕰️  Antigüedad máxima (años)"),
                    dcc.Slider(
                        id="filter-antiguedad",
                        min=0, max=ANTIGUEDAD_MAX, step=1, value=ANTIGUEDAD_MAX,
                        marks=_marcas_antiguedad,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"marginBottom": "18px"}),

                section_label("MODELO"),

                html.Div([
                    field_label("🎯  Especificación"),
                    dcc.RadioItems(
                        id="filter-modelo",
                        options=[
                            {"label": "  Simple · solo el área", "value": "simple"},
                            {"label": "  Múltiple · las cuatro variables",
                             "value": "multiple"},
                        ],
                        value="multiple",
                        labelStyle={"display": "block", "color": SIDEBAR_TEXT,
                                    "fontSize": "11.5px", "marginBottom": "5px",
                                    "cursor": "pointer"},
                        inputStyle={"marginRight": "6px"},
                    ),
                    html.Div(
                        "El modelo se reestima por completo sobre los "
                        "apartamentos filtrados: no se ocultan puntos, se vuelve "
                        "a ajustar la regresión por mínimos cuadrados.",
                        style={"color": SIDEBAR_MUTED, "fontSize": "10px",
                               "lineHeight": "1.45", "marginTop": "10px"},
                    ),
                ], style=dd_container_style),

                html.Button(
                    "✕  Limpiar filtros",
                    id="btn-clear",
                    n_clicks=0,
                    style={
                        "width": "100%", "padding": "8px 0",
                        "backgroundColor": "transparent", "color": SIDEBAR_MUTED,
                        "border": "1px solid rgba(255,255,255,0.12)",
                        "borderRadius": "6px", "cursor": "pointer",
                        "fontSize": "11px", "fontWeight": "600",
                        "letterSpacing": "0.4px", "marginTop": "4px",
                        "fontFamily": FONT, "transition": "all 0.2s",
                    },
                ),
            ],
            style={
                "padding": "20px 18px",
                "borderBottom": "1px solid rgba(255,255,255,0.1)",
                "flexShrink": "0",
            },
        ),

        # --- Nota metodológica -----------------------------------------------
        html.Div(
            [
                section_label("NOTA"),
                html.Div(
                    [
                        html.Span("El modelo múltiple usa "),
                        html.Span(f"{len(PREDICTORAS)} variables",
                                  style={"color": SIDEBAR_TEXT,
                                         "fontWeight": "600"}),
                        html.Span(": área, habitaciones, antigüedad y estrato. "
                                  "Al filtrar por un solo estrato esa columna se "
                                  "vuelve constante y se descarta del ajuste, "
                                  "porque una variable sin variación no puede "
                                  "explicar nada."),
                    ],
                    style={"color": SIDEBAR_MUTED, "fontSize": "10.5px",
                           "lineHeight": "1.5"},
                ),
            ],
            style={"padding": "18px",
                   "borderBottom": "1px solid rgba(255,255,255,0.1)",
                   "flexShrink": "0"},
        ),

        # --- Autor -----------------------------------------------------------
        html.Div(
            html.Div(
                [
                    html.Img(src=AUTHOR_SRC, style={
                        "width": "46px", "height": "46px", "borderRadius": "50%",
                        "objectFit": "cover", "border": f"2px solid {GOLD}",
                        "marginRight": "10px", "flexShrink": "0",
                    }) if AUTHOR_SRC else html.Div(style={
                        "width": "46px", "height": "46px", "borderRadius": "50%",
                        "backgroundColor": GOLD, "marginRight": "10px",
                        "flexShrink": "0",
                    }),
                    html.Div([
                        html.P("Andy Rubiano", style={
                            "color": SIDEBAR_TEXT, "margin": "0",
                            "fontSize": "13px", "fontWeight": "600"}),
                        html.P("Análisis de regresión", style={
                            "color": SIDEBAR_MUTED, "margin": "0",
                            "fontSize": "11px"}),
                    ]),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            style={
                "padding": "14px 18px",
                "borderTop": "1px solid rgba(255,255,255,0.1)",
                "marginTop": "auto", "flexShrink": "0",
            },
        ),
    ],
    style={
        # flexShrink 0 evita que el lienzo, al crecer, le robe ancho.
        "width": SIDEBAR_WIDTH, "minWidth": SIDEBAR_WIDTH, "flexShrink": "0",
        "backgroundColor": SIDEBAR_BG, "display": "flex",
        "flexDirection": "column", "height": "100%", "overflowY": "auto",
        "overflowX": "hidden", "fontFamily": FONT, "boxSizing": "border-box",
    },
)


encabezado = html.Div(
    [
        html.H1("ANÁLISIS DE REGRESIÓN DEL PRECIO DE VIVIENDA USADA", style={
            "margin": "0", "fontSize": "21px", "fontWeight": "800",
            "color": TITLE_COLOR, "letterSpacing": "0.7px",
        }),
        html.P(
            f"Regresión lineal simple y múltiple sobre {len(df)} apartamentos "
            "de Bogotá  ·  Por Andy Rubiano  ·  Universidad de La Salle",
            style={"color": TEXT_MUTED, "margin": "5px 0 0 0",
                   "fontSize": "12.5px"},
        ),
    ],
    style={
        "backgroundColor": CARD_BG, "padding": "20px 30px",
        "borderBottom": f"1px solid {CARD_BORDER}",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.05)", "fontFamily": FONT,
        "position": "sticky", "top": "0", "zIndex": "20",
    },
)


aviso_muestra = html.Div(id="aviso-muestra", style={"display": "none"})


tabla_coeficientes = dash_table.DataTable(
    id="table-coeficientes",
    columns=[
        {"name": "Término", "id": "Termino"},
        {"name": "Coeficiente", "id": "Coeficiente", "type": "numeric"},
        {"name": "Error estándar", "id": "ErrorEstandar", "type": "numeric"},
        {"name": "t", "id": "T", "type": "numeric"},
        {"name": "p-valor", "id": "PValor"},
        {"name": "IC 95 %", "id": "IC"},
        {"name": "¿Significativo?", "id": "Significativo"},
    ],
    style_header={
        "backgroundColor": "#F8FAFC", "color": TITLE_COLOR, "fontWeight": "700",
        "border": f"1px solid {CARD_BORDER}", "fontSize": "10.5px",
        "padding": "10px 10px", "fontFamily": FONT, "whiteSpace": "normal",
        "height": "auto",
    },
    style_cell={
        "backgroundColor": CARD_BG, "color": TEXT_COLOR,
        "border": f"1px solid {CARD_BORDER}", "fontSize": "11px",
        "padding": "9px 10px", "textAlign": "right", "fontFamily": FONT,
    },
    style_cell_conditional=[
        {"if": {"column_id": "Termino"}, "textAlign": "left",
         "minWidth": "210px", "fontWeight": "600"},
        {"if": {"column_id": "Significativo"}, "textAlign": "center"},
    ],
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#F8FAFC"},
        {"if": {"filter_query": "{Significativo} = 'sí'",
                "column_id": "Significativo"},
         "color": "#1B7F4B", "fontWeight": "700"},
        {"if": {"filter_query": "{Significativo} = 'no'",
                "column_id": "Significativo"},
         "color": TEXT_MUTED},
    ],
    style_table={"overflowX": "auto"},
)


lienzo = html.Div(
    [
        aviso_muestra,

        # --- Indicadores -----------------------------------------------------
        html.Div(
            [
                kpi_card("🏠", "", "Apartamentos analizados", card_id="kpi-n",
                         hint="Número de apartamentos que quedan tras aplicar "
                              "los filtros"),
                kpi_card("🎯", "", "R² del modelo", card_id="kpi-r2",
                         hint="Proporción de la variabilidad del precio que "
                              "explica la especificación elegida"),
                kpi_card("📏", "", "RMSE", card_id="kpi-rmse",
                         hint="Raíz del error cuadrático medio, en millones de "
                              "pesos"),
                kpi_card("💵", "", "Error medio", card_id="kpi-mae",
                         hint="Error absoluto medio y su peso sobre el precio "
                              "promedio del subconjunto"),
                kpi_card("🏙️", "", "Precio promedio", card_id="kpi-precio",
                         hint="Precio medio de los apartamentos filtrados"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                "gap": "14px", "marginBottom": "18px",
            },
        ),

        # --- Ajuste y efectos ------------------------------------------------
        html.Div(
            [
                card(dcc.Graph(id="fig-ajuste", config=GRAPH_CONFIG_FULL,
                               style={"height": "500px"}), flex=62,
                     min_w="420px"),
                card(dcc.Graph(id="fig-efectos", config=GRAPH_CONFIG,
                               style={"height": "500px"}), flex=38,
                     min_w="300px"),
            ],
            style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                   "flexWrap": "wrap"},
        ),

        # --- Diagnóstico -----------------------------------------------------
        html.Div(
            [
                card(dcc.Graph(id="fig-observado", config=GRAPH_CONFIG,
                               style={"height": "400px"}), flex=50,
                     min_w="330px"),
                card(dcc.Graph(id="fig-residuos", config=GRAPH_CONFIG,
                               style={"height": "400px"}), flex=50,
                     min_w="330px"),
            ],
            style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                   "flexWrap": "wrap"},
        ),

        # --- Tabla de coeficientes -------------------------------------------
        card(
            [
                html.H3("Coeficientes estimados sobre la selección actual",
                        style={
                            "color": TITLE_COLOR, "fontSize": "14px",
                            "fontWeight": "700", "margin": "0 0 4px 0",
                            "fontFamily": FONT,
                        }),
                html.P(
                    "Cada coeficiente mide el efecto de su variable sobre el "
                    "precio en millones de pesos, manteniendo constantes las "
                    "demás. Un término es significativo cuando su intervalo de "
                    "confianza no contiene al cero.",
                    style={"color": TEXT_MUTED, "fontSize": "11px",
                           "margin": "0 0 14px 0", "fontFamily": FONT},
                ),
                tabla_coeficientes,
            ],
            padding=True,
        ),

        # --- Pie -------------------------------------------------------------
        html.Div(
            html.P(
                f"Datos simulados con semilla fija (n = {len(df)})  ·  "
                "Mínimos cuadrados ordinarios con statsmodels, verificados en R "
                "con lm()  ·  Construido con Python, Dash y Plotly",
                style={"color": TEXT_MUTED, "textAlign": "center",
                       "fontSize": "11px", "margin": "0", "fontFamily": FONT},
            ),
            style={"borderTop": f"1px solid {CARD_BORDER}",
                   "padding": "16px 0 6px", "marginTop": "18px"},
        ),
    ],
    style={"padding": "22px 30px"},
)


layout = html.Div(
    [
        sidebar,
        # El scroll vive aquí, no en el documento: por eso la barra lateral no
        # necesita position fixed y el encabezado puede quedarse pegado arriba.
        html.Div(
            [encabezado, lienzo],
            id="lienzo-principal",
            style={
                "flex": "1", "height": "100%", "overflowY": "auto",
                "overflowX": "hidden", "backgroundColor": MAIN_BG,
                "fontFamily": FONT,
            },
        ),
    ],
    style={"display": "flex", "height": "100%", "overflow": "hidden",
           "fontFamily": FONT},
)
