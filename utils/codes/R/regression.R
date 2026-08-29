#' Fase 5 - Los mismos modelos en R con lm() y ggplot2.
#'
#' Reestima con lm() la regresion simple y la multiple de la Fase 2 y compara
#' los coeficientes contra los que estimo statsmodels. Que dos implementaciones
#' independientes lleguen al mismo numero es la mejor prueba de que el
#' resultado no depende de la herramienta.
#'
#' Produce ademas las tres figuras en ggplot2 del informe.
#'
#' Ejecutar desde la raiz del proyecto:
#'   Rscript utils/codes/R/regression.R
#' o abrirlo en RStudio y pulsar Source.

library(ggplot2)

# 0. Rutas
# R no tiene un equivalente de __file__, asi que la ruta del script se busca en
# los dos modos de ejecucion posibles: Rscript (argumento --file=) y el boton
# Source de RStudio. Si ninguno funciona se usa el directorio de trabajo.
argumentos <- commandArgs(trailingOnly = FALSE)
archivo <- sub("^--file=", "", grep("^--file=", argumentos, value = TRUE))
if (length(archivo) == 0 && requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()) {
  archivo <- rstudioapi::getSourceEditorContext()$path
}
raiz <- if (length(archivo) > 0 && nzchar(archivo)) {
  # utils/codes/R/regression.R -> utils/codes/R -> utils/codes -> utils -> raiz
  dirname(dirname(dirname(dirname(normalizePath(archivo)))))
} else {
  getwd()
}

dataset <- file.path(raiz, "data", "dataset", "viviendas.csv")
tablas <- file.path(raiz, "data", "processed")
figuras <- file.path(raiz, "public", "assets", "images", "figures", "r")

if (!file.exists(dataset)) {
  stop(sprintf("No se encontro %s. Ejecuta antes dataset.py.", dataset))
}
if (!dir.exists(figuras)) dir.create(figuras, recursive = TRUE)

cat(sprintf("Raiz del proyecto: %s\n\n", raiz))

df <- read.csv(dataset)
azules <- c("3" = "#a6bddb", "4" = "#4292c6", "5" = "#08519c")
naranja <- "#d95f02"

tema <- theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(colour = "grey35"),
        legend.position = "bottom")

# 1. Regresion lineal simple: precio ~ area
simple <- lm(precio_millones_cop ~ area_m2, data = df)
cat("=== Modelo simple ===\n")
print(summary(simple))

# 2. Regresion lineal multiple
multiple <- lm(precio_millones_cop ~ area_m2 + habitaciones +
                 antiguedad_anios + estrato, data = df)
cat("\n=== Modelo multiple ===\n")
print(summary(multiple))

# El contraste F entre los dos modelos responde si las tres variables anadidas
# aportan lo suficiente para justificar los grados de libertad que gastan.
cat("\n=== Contraste F: el modelo multiple frente al simple ===\n")
print(anova(simple, multiple))

# 3. Coeficientes y comparacion con Python
resumen <- summary(multiple)$coefficients
intervalos <- confint(multiple)
coeficientes_r <- data.frame(
  termino = rownames(resumen),
  coeficiente = round(resumen[, "Estimate"], 3),
  error_estandar = round(resumen[, "Std. Error"], 3),
  estadistico_t = round(resumen[, "t value"], 2),
  p_valor = format(resumen[, "Pr(>|t|)"], digits = 3, scientific = TRUE),
  ic95_inferior = round(intervalos[, 1], 3),
  ic95_superior = round(intervalos[, 2], 3),
  row.names = NULL
)
write.csv(coeficientes_r, file.path(tablas, "regresion_multiple_r.csv"),
          row.names = FALSE)
cat("\nCoeficientes estimados con lm()\n")
print(coeficientes_r, row.names = FALSE)

# Verificacion: la diferencia con statsmodels deberia ser cero hasta el
# redondeo. Si alguna vez no lo fuera, es que los dos modelos no son el mismo.
ruta_python <- file.path(tablas, "regresion_multiple.csv")
if (file.exists(ruta_python)) {
  coeficientes_py <- read.csv(ruta_python)
  verificacion <- data.frame(
    termino = coeficientes_r$termino,
    python_statsmodels = coeficientes_py$coeficiente,
    r_lm = coeficientes_r$coeficiente,
    diferencia = round(coeficientes_py$coeficiente - coeficientes_r$coeficiente, 6)
  )
  write.csv(verificacion, file.path(tablas, "verificacion_python_r.csv"),
            row.names = FALSE)
  cat("\nVerificacion cruzada Python vs R\n")
  print(verificacion, row.names = FALSE)
  cat(sprintf("\nDiferencia maxima entre las dos implementaciones: %.6f\n",
              max(abs(verificacion$diferencia))))
}

r2_simple <- summary(simple)$r.squared
r2_multiple <- summary(multiple)$r.squared
cat(sprintf("\nR2 simple = %.4f | R2 multiple = %.4f | ganancia = %.1f p.p.\n",
            r2_simple, r2_multiple, (r2_multiple - r2_simple) * 100))

# 4. Figuras con ggplot2
df$estrato_f <- factor(df$estrato)
df$precio_estimado <- round(fitted(multiple), 1)

# Figura 1: el ajuste simple con su banda de confianza. geom_smooth ajusta el
# mismo lm() por dentro, asi que la recta dibujada es exactamente la estimada.
g1 <- ggplot(df, aes(area_m2, precio_millones_cop)) +
  geom_point(colour = "#2b8cbe", alpha = 0.8, size = 2) +
  geom_smooth(method = "lm", formula = y ~ x, colour = naranja,
              fill = naranja, alpha = 0.2) +
  labs(title = "Regresion simple en R: precio frente a area",
       subtitle = sprintf("lm(precio ~ area) - R2 = %.4f - pendiente = %.2f millones por m2",
                          r2_simple, coef(simple)[2]),
       x = "Area (m2)", y = "Precio (millones de COP)") +
  tema
ggsave(file.path(figuras, "ggplot_ajuste_simple.png"), g1,
       width = 7.5, height = 4.5, dpi = 150)

# Figura 2: un panel por estrato. Las tres rectas tienen pendiente parecida y
# alturas distintas, que es justo lo que el modelo multiple supone.
g2 <- ggplot(df, aes(area_m2, precio_millones_cop, colour = estrato_f)) +
  geom_point(alpha = 0.85, size = 2) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, alpha = 0.15,
              aes(fill = estrato_f)) +
  facet_wrap(~ estrato_f, labeller = labeller(estrato_f = function(x) paste("Estrato", x))) +
  scale_colour_manual(values = azules, name = "Estrato") +
  scale_fill_manual(values = azules, name = "Estrato") +
  labs(title = "Una regresion por estrato",
       subtitle = "Misma pendiente, distinta altura: el estrato desplaza el precio hacia arriba",
       x = "Area (m2)", y = "Precio (millones de COP)") +
  tema
ggsave(file.path(figuras, "ggplot_facetas_estrato.png"), g2,
       width = 9.5, height = 4.2, dpi = 150)

# Figura 3: precio real frente al estimado. La diagonal es la prediccion
# perfecta; cuanto mas pegados a ella esten los puntos, mejor el modelo.
g3 <- ggplot(df, aes(precio_estimado, precio_millones_cop, colour = estrato_f)) +
  geom_abline(slope = 1, intercept = 0, colour = naranja, linetype = "dashed",
              linewidth = 0.9) +
  geom_point(alpha = 0.85, size = 2) +
  scale_colour_manual(values = azules, name = "Estrato") +
  coord_equal() +
  labs(title = "Modelo multiple: precio real frente al estimado",
       subtitle = sprintf("lm con cuatro variables - R2 = %.4f - error tipico = %.1f millones",
                          r2_multiple, summary(multiple)$sigma),
       x = "Precio estimado (millones de COP)",
       y = "Precio real (millones de COP)") +
  tema
ggsave(file.path(figuras, "ggplot_real_vs_estimado.png"), g3,
       width = 6.5, height = 5.5, dpi = 150)

cat("\nOK - Fase 5: 2 tablas en data/processed y 3 figuras en figures/r\n")
