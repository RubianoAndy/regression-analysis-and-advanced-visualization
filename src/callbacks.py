"""Lógica reactiva: un solo callback reestima el modelo y redibuja todo.

Se concentra el trabajo en un único callback porque las cinco tarjetas, las
cuatro figuras y la tabla dependen del **mismo** ajuste. Repartirlo en varios
obligaría a reestimar la regresión una vez por salida, o a guardar el modelo en
un ``dcc.Store``, y ninguna de las dos cosas compensa cuando el ajuste tarda
milisegundos.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State

from src.data import (
    ANTIGUEDAD_MAX, AREA_MAX, AREA_MIN, MINIMO_APARTAMENTOS, OBJETIVO,
    ajustar, banda_ajuste, filtrar,
)
from src.theme import (
    BASE_LAYOUT, CARD_BORDER, GRIS, NARANJA, TEXT_MUTED, TITLE_COLOR,
    color_estrato, title_cfg,
)

REJILLA = dict(gridcolor="#EEF2F7", zeroline=False)


def _formato_p(p):
    """p-valor legible: notación decimal si es grande, científica si es diminuto."""
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def figura_vacia(mensaje):
    """Figura con un mensaje centrado, para cuando no hay ajuste posible."""
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False,
                       font=dict(size=13, color=TEXT_MUTED),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(**BASE_LAYOUT,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _fig_ajuste(res, sub):
    """Precio frente a área, con la recta reestimada y su banda de confianza."""
    fig = go.Figure()

    malla, pred = banda_ajuste(res, sub)
    fig.add_trace(go.Scatter(
        x=np.concatenate([malla, malla[::-1]]),
        y=np.concatenate([pred["mean_ci_upper"], pred["mean_ci_lower"][::-1]]),
        fill="toself", fillcolor="rgba(217,95,2,0.18)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    for estrato in sorted(sub["estrato"].unique()):
        g = sub[sub["estrato"] == estrato]
        fig.add_trace(go.Scatter(
            x=g["area_m2"], y=g[OBJETIVO], mode="markers",
            name=f"Estrato {estrato}",
            marker=dict(size=8, color=color_estrato(estrato),
                        line=dict(width=0.5, color="white")),
            customdata=g[["inmueble_id", "antiguedad_anios", "habitaciones"]],
            hovertemplate=("<b>%{customdata[0]}</b><br>Área: %{x:.0f} m²<br>"
                           "Precio: %{y:.0f} M COP<br>"
                           "Antigüedad: %{customdata[1]} años<br>"
                           "Habitaciones: %{customdata[2]}<extra></extra>"),
        ))

    fig.add_trace(go.Scatter(
        x=malla, y=pred["mean"], mode="lines", name="Recta reestimada",
        line=dict(color=NARANJA, width=2.6),
        hovertemplate="Área: %{x:.0f} m²<br>Precio esperado: %{y:.0f} M COP"
                      "<extra></extra>",
    ))

    pendiente = res["modelo"].params.get("area_m2", float("nan"))
    fig.update_layout(
        **BASE_LAYOUT,
        title=title_cfg(
            "Ajuste por mínimos cuadrados sobre la selección",
            f"R² = {res['r2']:.4f}  ·  {res['n']} apartamentos  ·  "
            f"cada m² vale {pendiente:.2f} millones",
        ),
        xaxis=dict(title="Área (m²)", **REJILLA),
        yaxis=dict(title="Precio (millones COP)", **REJILLA),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        hovermode="closest",
    )
    return fig


def _fig_efectos(res):
    """Coeficientes con su intervalo de confianza al 95 %."""
    coefs = res["coeficientes"]
    coefs = coefs[coefs["variable"] != "Intercept"]
    if coefs.empty:
        return figura_vacia("El modelo solo conserva el intercepto")

    coefs = coefs.iloc[::-1]
    significativo = coefs["p_valor"] < 0.05
    colores = [TITLE_COLOR if s else GRIS for s in significativo]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs["coeficiente"], y=coefs["termino"], mode="markers",
        marker=dict(size=12, color=colores),
        error_x=dict(
            type="data", symmetric=False,
            array=coefs["ic_superior"] - coefs["coeficiente"],
            arrayminus=coefs["coeficiente"] - coefs["ic_inferior"],
            color="#6baed6", thickness=2.4, width=6,
        ),
        customdata=np.stack([coefs["p_valor"], coefs["ic_inferior"],
                             coefs["ic_superior"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Efecto: %{x:+.2f} M COP<br>"
                       "IC 95 %%: [%{customdata[1]:.2f} ; %{customdata[2]:.2f}]"
                       "<br>p-valor: %{customdata[0]:.2e}<extra></extra>"),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=NARANJA, width=1.4, dash="dash"))

    fig.update_layout(
        **BASE_LAYOUT,
        title=title_cfg(
            "Efecto de cada variable sobre el precio",
            "Azul si el intervalo de confianza excluye el cero",
        ),
        xaxis=dict(title="Cambio en el precio (millones COP)", **REJILLA),
        yaxis=dict(**REJILLA),
    )
    return fig


def _fig_observado(res, sub):
    """Precio real frente al estimado, contra la diagonal de acierto perfecto."""
    y = sub[OBJETIVO]
    fig = go.Figure()

    limites = [float(min(y.min(), res["ajustados"].min())) * 0.92,
               float(max(y.max(), res["ajustados"].max())) * 1.05]
    fig.add_trace(go.Scatter(
        x=limites, y=limites, mode="lines", name="Predicción perfecta",
        line=dict(color=NARANJA, width=1.8, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))

    for estrato in sorted(sub["estrato"].unique()):
        m = (sub["estrato"] == estrato).to_numpy()
        fig.add_trace(go.Scatter(
            x=res["ajustados"][m], y=y[m], mode="markers",
            name=f"Estrato {estrato}", showlegend=False,
            marker=dict(size=7, color=color_estrato(estrato),
                        line=dict(width=0.5, color="white")),
            hovertemplate="Estimado: %{x:.0f} M COP<br>"
                          "Real: %{y:.0f} M COP<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=title_cfg("Precio real frente al estimado",
                        "Cuanto más pegados a la diagonal, mejor el modelo"),
        xaxis=dict(title="Precio estimado (millones COP)", **REJILLA),
        yaxis=dict(title="Precio real (millones COP)", **REJILLA),
    )
    return fig


def _fig_residuos(res, sub):
    """Errores frente al valor estimado: deben repartirse en torno a cero."""
    fig = go.Figure()
    for estrato in sorted(sub["estrato"].unique()):
        m = (sub["estrato"] == estrato).to_numpy()
        fig.add_trace(go.Scatter(
            x=res["ajustados"][m], y=res["residuos"][m], mode="markers",
            name=f"Estrato {estrato}", showlegend=False,
            marker=dict(size=7, color=color_estrato(estrato),
                        line=dict(width=0.5, color="white")),
            hovertemplate="Estimado: %{x:.0f} M COP<br>"
                          "Error: %{y:+.0f} M COP<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=NARANJA, width=1.6))

    fig.update_layout(
        **BASE_LAYOUT,
        title=title_cfg(
            "Errores de estimación",
            f"RMSE = {res['rmse']:.1f} millones  ·  sin patrón visible, el "
            "modelo captura la estructura",
        ),
        xaxis=dict(title="Precio estimado (millones COP)", **REJILLA),
        yaxis=dict(title="Error (millones COP)", **REJILLA),
    )
    return fig


def _aviso(texto, tono="info"):
    """Franja de aviso sobre el lienzo."""
    fondos = {"info": ("#FFF8E1", "#8A6D00"), "alerta": ("#FDECEA", "#A3241B")}
    fondo, color = fondos[tono]
    return {
        "display": "block", "backgroundColor": fondo, "color": color,
        "border": f"1px solid {CARD_BORDER}", "padding": "11px 16px",
        "marginBottom": "16px", "fontSize": "11.5px", "lineHeight": "1.5",
    }, texto


def register_callbacks(app):
    """Conecta los controles con el lienzo."""

    @app.callback(
        Output("filter-estrato", "value"),
        Output("filter-area", "value"),
        Output("filter-antiguedad", "value"),
        Output("filter-modelo", "value"),
        Input("btn-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def limpiar(_):
        return [], [AREA_MIN, AREA_MAX], ANTIGUEDAD_MAX, "multiple"

    @app.callback(
        Output("kpi-n", "children"),
        Output("kpi-r2", "children"),
        Output("kpi-rmse", "children"),
        Output("kpi-mae", "children"),
        Output("kpi-precio", "children"),
        Output("fig-ajuste", "figure"),
        Output("fig-efectos", "figure"),
        Output("fig-observado", "figure"),
        Output("fig-residuos", "figure"),
        Output("table-coeficientes", "data"),
        Output("aviso-muestra", "style"),
        Output("aviso-muestra", "children"),
        Input("filter-estrato", "value"),
        Input("filter-area", "value"),
        Input("filter-antiguedad", "value"),
        Input("filter-modelo", "value"),
    )
    def actualizar(estratos, rango_area, antiguedad_max, especificacion):
        sub = filtrar(estratos, rango_area, antiguedad_max)
        res = ajustar(sub, especificacion)

        if res is None:
            mensaje = (
                f"La selección deja {len(sub)} apartamentos, por debajo del "
                f"mínimo de {MINIMO_APARTAMENTOS} necesario para estimar el "
                "modelo con intervalos de confianza interpretables. Amplíe los "
                "filtros."
            )
            estilo, texto = _aviso(mensaje, tono="alerta")
            vacia = figura_vacia("Muestra insuficiente para ajustar el modelo")
            return ("—", "—", "—", "—", "—",
                    vacia, vacia, vacia, vacia, [], estilo, texto)

        coefs = res["coeficientes"]
        tabla = [
            {
                "Termino": r.termino if r.variable != "Intercept" else "Intercepto",
                "Coeficiente": round(r.coeficiente, 3),
                "ErrorEstandar": round(r.error_estandar, 3),
                "T": round(r.estadistico_t, 2),
                "PValor": _formato_p(r.p_valor),
                "IC": f"[{r.ic_inferior:.2f} ; {r.ic_superior:.2f}]",
                "Significativo": "sí" if r.p_valor < 0.05 else "no",
            }
            for r in coefs.itertuples()
        ]

        if res["descartadas"]:
            nombres = ", ".join(res["descartadas"])
            estilo, texto = _aviso(
                f"Los filtros dejan sin variación estas columnas: {nombres}. "
                "Se descartaron del ajuste, porque una variable constante no "
                "puede explicar diferencias de precio.",
            )
        else:
            estilo, texto = {"display": "none"}, ""

        porcentaje = res["mae"] / res["precio_medio"] * 100
        return (
            f"{res['n']}",
            f"{res['r2']:.3f}",
            f"{res['rmse']:.1f}",
            f"{res['mae']:.1f}  ({porcentaje:.1f} %)",
            f"{res['precio_medio']:.0f} M",
            _fig_ajuste(res, sub),
            _fig_efectos(res),
            _fig_observado(res, sub),
            _fig_residuos(res, sub),
            tabla,
            estilo,
            texto,
        )
