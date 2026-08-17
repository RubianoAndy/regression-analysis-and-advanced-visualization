#' Actividad 4 - Fase 5: verificacion cruzada de la regresion en R.
#'
#' Reestima con lm() los mismos modelos que la Fase 2 ajusto con statsmodels
#' —simple, multiple con interaccion y log-log—, contrasta los modelos
#' anidados con anova() y compara coeficiente a coeficiente contra las tablas
#' que escribio Python. Las figuras se reparten a proposito entre los dos
#' sistemas graficos de R: ggplot2 para las vistas del ajuste y la graficacion
#' base para el diagnostico canonico plot(modelo).
#'
#' Ejecutar desde la raiz del proyecto. Escribe las tablas en data/processed y
#' las imagenes en public/assets/images/figures/r/regression/.

library(ggplot2)

data_path <- "data/dataset/consumo_energia.csv"
processed_dir <- file.path("data", "processed")
figures_dir <- file.path("public", "assets", "images", "figures", "r",
                         "regression")
if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
}

sector_order <- c("Residencial", "Comercial", "Industrial")
sector_colors <- c(Residencial = "#a6bddb", Comercial = "#74a9cf",
                   Industrial = "#2b8cbe")
accent <- "#d95f02"

df <- read.csv(data_path)
df$sector <- factor(df$sector, levels = sector_order)
df$tarifa_cop_kwh <- df$costo_miles_cop * 1000 / df$consumo_kwh
n <- nrow(df)

#' 1. ESTIMACION DE LOS TRES MODELOS.
#'
#' El operador * de la formula de R expande igual que el de patsy: efectos
#' principales mas interaccion. Residencial es el nivel de referencia por ser
#' el primero del factor, de modo que los coeficientes coinciden uno a uno con
#' los de Python.
m1 <- lm(costo_miles_cop ~ consumo_kwh, data = df)
m3 <- lm(costo_miles_cop ~ consumo_kwh * sector, data = df)
m4 <- lm(log(costo_miles_cop) ~ log(consumo_kwh) + sector, data = df)

cat("=== M1 - regresion lineal simple ===\n")
print(summary(m1))
cat("\n=== M3 - regresion multiple con interaccion ===\n")
print(summary(m3))
cat("\n=== M4 - modelo log-log ===\n")
print(summary(m4))

#' 2. CONTRASTE DE MODELOS ANIDADOS Y METRICAS COMPARABLES.
cat("\n=== Contraste F entre modelos anidados ===\n")
m2 <- lm(costo_miles_cop ~ consumo_kwh + sector, data = df)
print(anova(m1, m2, m3))

rmse <- function(observado, predicho) sqrt(mean((observado - predicho)^2))
factor_duan <- mean(exp(residuals(m4)))
pred_m4 <- exp(fitted(m4)) * factor_duan
jacobiano <- sum(log(df$costo_miles_cop))

metricas <- data.frame(
  modelo = c("M1 - Simple", "M2 - Aditivo", "M3 - Interaccion", "M4 - Log-log"),
  r2 = round(c(summary(m1)$r.squared, summary(m2)$r.squared,
               summary(m3)$r.squared, summary(m4)$r.squared), 4),
  r2_ajustado = round(c(summary(m1)$adj.r.squared, summary(m2)$adj.r.squared,
                        summary(m3)$adj.r.squared, summary(m4)$adj.r.squared), 4),
  aic = round(c(AIC(m1), AIC(m2), AIC(m3), AIC(m4) + 2 * jacobiano), 1),
  rmse = round(c(rmse(df$costo_miles_cop, fitted(m1)),
                 rmse(df$costo_miles_cop, fitted(m2)),
                 rmse(df$costo_miles_cop, fitted(m3)),
                 rmse(df$costo_miles_cop, pred_m4)), 2)
)
cat("\n=== Metricas de los cuatro modelos (R) ===\n")
print(metricas, row.names = FALSE)
cat("Nota: el AIC de R supera en 2,0 al de statsmodels porque R cuenta la\n",
    "varianza residual como un parametro mas. La diferencia es constante y\n",
    "no altera el orden de los modelos.\n")
write.csv(metricas, file.path(processed_dir, "comparacion_modelos_r.csv"),
          row.names = FALSE)

#' 3. COEFICIENTES Y VERIFICACION CRUZADA CONTRA PYTHON.
#'
#' Se comparan los seis coeficientes de M3 con los que dejo statsmodels en
#' regresion_multiple.csv. La coincidencia debe darse en todas las cifras
#' reportadas: ambos resuelven el mismo sistema de ecuaciones normales.
coef_m3 <- data.frame(
  termino = names(coef(m3)),
  coeficiente = round(unname(coef(m3)), 4),
  error_std = round(unname(summary(m3)$coefficients[, 2]), 4),
  estadistico_t = round(unname(summary(m3)$coefficients[, 3]), 2),
  p_valor = formatC(unname(summary(m3)$coefficients[, 4]), format = "e",
                    digits = 2),
  ic95_inferior = round(unname(confint(m3)[, 1]), 4),
  ic95_superior = round(unname(confint(m3)[, 2]), 4)
)
cat("\n=== Coeficientes de M3 en R ===\n")
print(coef_m3, row.names = FALSE)
write.csv(coef_m3, file.path(processed_dir, "regresion_multiple_r.csv"),
          row.names = FALSE)

#' R y patsy ordenan la matriz de diseno de forma distinta —R agrupa primero
#' las variables continuas y patsy primero las categoricas—, asi que la
#' comparacion se hace emparejando por nombre de termino y no por posicion.
orden_python <- c("(Intercept)", "sectorComercial", "sectorIndustrial",
                  "consumo_kwh", "consumo_kwh:sectorComercial",
                  "consumo_kwh:sectorIndustrial")
python_path <- file.path(processed_dir, "regresion_multiple.csv")
if (file.exists(python_path)) {
  coef_python <- read.csv(python_path)
  coef_r_ordenado <- coef(m3)[orden_python]
  comparacion <- data.frame(
    termino = coef_python$termino,
    termino_r = orden_python,
    python = coef_python$coeficiente,
    r = round(unname(coef_r_ordenado), 4),
    diferencia_absoluta = abs(coef_python$coeficiente -
                                round(unname(coef_r_ordenado), 4))
  )
  cat("\n=== Verificacion cruzada Python vs R (coeficientes de M3) ===\n")
  print(comparacion, row.names = FALSE)
  cat(sprintf("Diferencia maxima: %.10f\n",
              max(comparacion$diferencia_absoluta)))
  write.csv(comparacion, file.path(processed_dir, "verificacion_cruzada.csv"),
            row.names = FALSE)
} else {
  cat("\nNo se encontro regresion_multiple.csv: ejecuta antes la Fase 2.\n")
}

#' Pendiente por sector, que es la tarifa implicita de cada grupo.
pendientes <- c(
  Residencial = unname(coef(m3)["consumo_kwh"]),
  Comercial = unname(coef(m3)["consumo_kwh"] +
                       coef(m3)["consumo_kwh:sectorComercial"]),
  Industrial = unname(coef(m3)["consumo_kwh"] +
                        coef(m3)["consumo_kwh:sectorIndustrial"])
)
cat("\n=== Tarifa implicita por sector (COP/kWh) ===\n")
print(round(pendientes * 1000, 1))

#' 4. FIGURAS CON GGPLOT2.
#'
#' La gramatica de graficos declara el ajuste como una capa mas: geom_smooth
#' con method = "lm" reestima la regresion dentro de cada grupo y dibuja su
#' banda de confianza sin necesidad de calcularla aparte.

tema_informe <- theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top")

#' Ajuste global frente a ajuste por sector, en un mismo lienzo.
g1 <- ggplot(df, aes(x = consumo_kwh, y = costo_miles_cop)) +
  geom_point(aes(color = sector), size = 2, alpha = 0.9) +
  geom_smooth(method = "lm", formula = y ~ x, color = accent,
              linewidth = 0.9, se = TRUE) +
  scale_color_manual(values = sector_colors, name = "Sector") +
  labs(title = "Regresión lineal simple del costo sobre el consumo (R)",
       subtitle = sprintf("R2 = %.4f, pendiente = %.1f COP/kWh",
                          summary(m1)$r.squared,
                          coef(m1)["consumo_kwh"] * 1000),
       x = "Consumo (kWh/mes)", y = "Costo facturado (miles de COP)") +
  tema_informe
ggsave(file.path(figures_dir, "ggplot_ajuste_simple.png"), g1,
       width = 7.2, height = 4.2, dpi = 150, type = "cairo")

#' Una faceta por sector con escalas libres: la relacion dentro de cada grupo
#' sin que la escala del sector Industrial aplaste a los demas.
etiquetas_tarifa <- setNames(
  sprintf("%s - %.0f COP/kWh", sector_order, pendientes[sector_order] * 1000),
  sector_order)
g2 <- ggplot(df, aes(x = consumo_kwh, y = costo_miles_cop, color = sector)) +
  geom_point(size = 2, alpha = 0.9) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, color = accent,
              linewidth = 0.9) +
  facet_wrap(~ sector, scales = "free",
             labeller = labeller(sector = etiquetas_tarifa)) +
  scale_color_manual(values = sector_colors, guide = "none") +
  labs(title = "Una regresión por sector: cada pendiente es una tarifa (R)",
       x = "Consumo (kWh/mes)", y = "Costo facturado (miles de COP)") +
  tema_informe
ggsave(file.path(figures_dir, "ggplot_ajuste_por_sector.png"), g2,
       width = 9.6, height = 3.8, dpi = 150, type = "cairo")

#' Residuos de M1 y M3 enfrentados: la misma comparacion que en Python, para
#' comprobar que el sesgo por sector no es un artefacto de una libreria.
residuos <- rbind(
  data.frame(modelo = "M1 · simple", sector = df$sector,
             ajustado = fitted(m1), residuo = residuals(m1)),
  data.frame(modelo = "M3 · con interacción", sector = df$sector,
             ajustado = fitted(m3), residuo = residuals(m3))
)
g3 <- ggplot(residuos, aes(x = sector, y = residuo, fill = sector)) +
  geom_hline(yintercept = 0, color = accent, linewidth = 0.7) +
  geom_boxplot(alpha = 0.95, outlier.size = 1.2) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3,
               color = accent) +
  facet_wrap(~ modelo) +
  scale_fill_manual(values = sector_colors, guide = "none") +
  labs(title = "El sesgo por sector desaparece al incluir la interacción (R)",
       subtitle = "El rombo marca el residuo medio de cada sector",
       x = "Sector", y = "Residuo (miles de COP)") +
  tema_informe
ggsave(file.path(figures_dir, "ggplot_residuos_por_sector.png"), g3,
       width = 9.0, height = 4.0, dpi = 150, type = "cairo")

#' Coeficientes con su intervalo de confianza, separando interceptos de
#' pendientes porque estan en escalas distintas.
ic <- as.data.frame(confint(m3))
names(ic) <- c("inferior", "superior")
ic$termino <- rownames(ic)
ic$coeficiente <- coef(m3)
ic$bloque <- ifelse(grepl("consumo_kwh", ic$termino),
                    "Pendientes (miles de COP por kWh)",
                    "Interceptos (miles de COP)")
ic$termino <- factor(ic$termino, levels = rev(ic$termino))
g4 <- ggplot(ic, aes(x = coeficiente, y = termino)) +
  geom_vline(xintercept = 0, color = accent, linetype = "dashed",
             linewidth = 0.7) +
  geom_pointrange(aes(xmin = inferior, xmax = superior), color = "#2b8cbe",
                  linewidth = 0.8, size = 0.5) +
  facet_wrap(~ bloque, scales = "free") +
  labs(title = "Coeficientes de M3 con intervalo de confianza al 95 % (R)",
       x = "Coeficiente", y = NULL) +
  tema_informe +
  theme(axis.text.y = element_text(size = 8))
ggsave(file.path(figures_dir, "ggplot_coeficientes.png"), g4,
       width = 10.0, height = 3.6, dpi = 150, type = "cairo")

#' 5. DIAGNOSTICO CANONICO CON LA GRAFICACION BASE.
#'
#' plot(modelo) devuelve las cuatro vistas clasicas del diagnostico de un lm
#' —residuos frente a ajustados, Q-Q normal, escala-localizacion y residuos
#' frente a apalancamiento con las curvas de Cook—, etiquetando ademas las
#' observaciones mas influyentes.
png(file.path(figures_dir, "base_diagnostico_m1.png"),
    width = 2100, height = 1500, res = 200, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(m1, col = "#2c7fb8", pch = 19, cex = 0.6)
dev.off()

png(file.path(figures_dir, "base_diagnostico_m3.png"),
    width = 2100, height = 1500, res = 200, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(m3, col = "#2c7fb8", pch = 19, cex = 0.6)
dev.off()

cat("\nOK - Fase 5: verificacion cruzada en R y figuras generadas\n")
