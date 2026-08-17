#' Actividad 2 - Verificacion cruzada en R del analisis estadistico.
#'
#' Lee el CSV generado por statistics.py, recalcula de forma independiente la
#' distribucion de frecuencias (Sturges), las medidas de tendencia central
#' (media, mediana, moda por clase modal interpolada) y de dispersion (rango,
#' varianza, desviacion estandar, coeficiente de variacion), y replica los
#' graficos basicos con la graficacion base, trazando la cuadricula en dos
#' pasadas para mantenerla detras de los datos.
#'
#' Escribe las imagenes en public/assets/images/figures/r/statistics/.

#' Rutas del proyecto (ejecutar desde la raiz del proyecto).
data_path <- "data/dataset/consumo_energia.csv"
figures_dir <- file.path("public", "assets", "images", "figures", "r",
                         "statistics")
if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
}

#' Carga de datos y vista rapida de su estructura.
df <- read.csv(data_path)
head(df)
str(df)

x <- df$consumo_kwh
n <- nrow(df)

#' 1. DISTRIBUCION DE FRECUENCIAS (regla de Sturges).
#'
#' Mismas k clases de igual amplitud que en Python; la tabla reporta fi, Fi,
#' hi (%) y Hi (%) para la verificacion cruzada.
k <- ceiling(1 + 3.322 * log10(n))
edges <- seq(min(x), max(x), length.out = k + 1)
fi <- hist(x, breaks = edges, plot = FALSE, include.lowest = TRUE)$counts
freq_table <- data.frame(
  lim_inferior = round(edges[-(k + 1)], 1),
  lim_superior = round(edges[-1], 1),
  marca_clase = round((edges[-(k + 1)] + edges[-1]) / 2, 1),
  fi = fi,
  Fi = cumsum(fi),
  hi_pct = round(fi / n * 100, 1),
  Hi_pct = round(cumsum(fi) / n * 100, 1)
)
print(freq_table)

#' Moda interpolada dentro de la clase modal: Mo = L + d1 / (d1 + d2) * w.
interpolated_mode <- function(values, bin_edges) {
  counts <- hist(values, breaks = bin_edges, plot = FALSE,
                 include.lowest = TRUE)$counts
  m <- which.max(counts)
  width <- bin_edges[m + 1] - bin_edges[m]
  d1 <- counts[m] - ifelse(m > 1, counts[m - 1], 0)
  d2 <- counts[m] - ifelse(m < length(counts), counts[m + 1], 0)
  if (d1 + d2 == 0) {
    return(bin_edges[m] + width / 2)
  }
  bin_edges[m] + d1 / (d1 + d2) * width
}

#' 2 y 3. TENDENCIA CENTRAL Y DISPERSION por sector y a nivel global.
#'
#' var() y sd() de R son muestrales (ddof = 1), igual que en pandas, por lo
#' que los valores deben coincidir digito a digito con central_tendency.csv y
#' dispersion.csv.
summary_stats <- function(values) {
  k_g <- ceiling(1 + 3.322 * log10(length(values)))
  edges_g <- seq(min(values), max(values), length.out = k_g + 1)
  c(
    n = length(values),
    media = round(mean(values), 1),
    mediana = round(median(values), 1),
    moda_interpolada = round(interpolated_mode(values, edges_g), 1),
    rango = round(max(values) - min(values), 1),
    varianza = round(var(values), 1),
    desv_std = round(sd(values), 1),
    cv_pct = round(sd(values) / mean(values) * 100, 1)
  )
}

groups <- c("Residencial", "Comercial", "Industrial")
stats_by_group <- t(sapply(groups,
                           function(s) summary_stats(x[df$sector == s])))
stats_global <- summary_stats(x)
print(rbind(stats_by_group, Global = stats_global))
cat("Moda del sector (variable nominal):",
    names(which.max(table(df$sector))), "\n")

#' 4. GRAFICOS BASICOS (replicas de las figuras de Python).
#'
#' Principios aplicados: titulo informativo, ejes con unidades, cuadricula
#' sutil detras de los datos, color funcional y etiquetas de datos.

#' Histograma con clases de Sturges y medidas de tendencia central.
#'
#' La primera pasada traza un histograma sin relleno para fijar ejes y
#' escala, la cuadricula se inserta detras y la segunda pasada dibuja las
#' barras encima.
png(file.path(figures_dir, "hist_sturges_central_tendency.png"),
    width = 1950, height = 1140, res = 300, type = "cairo")
par(mar = c(4, 4, 3, 1))
hist(x, breaks = edges, border = NA, col = NA, include.lowest = TRUE,
     main = sprintf("Distribución de frecuencias del consumo (R, %d clases)",
                    k),
     xlab = "Consumo (kWh/mes)", ylab = "Frecuencia absoluta (clientes)")
grid()
hist(x, breaks = edges, col = "#2c7fb8", border = "white",
     include.lowest = TRUE, add = TRUE)
global_mode <- interpolated_mode(x, edges)
abline(v = c(global_mode, median(x), mean(x)),
       col = "#d95f02", lty = c(3, 2, 1), lwd = 1.4)
legend("topright", cex = 0.75, col = "#d95f02", lty = c(3, 2, 1), lwd = 1.4,
       legend = c(sprintf("Moda = %.0f", global_mode),
                  sprintf("Mediana = %.0f", median(x)),
                  sprintf("Media = %.0f", mean(x))))
dev.off()

#' Poligono de frecuencias y ojiva en un mismo lienzo de dos paneles.
png(file.path(figures_dir, "freq_polygon_ogive.png"),
    width = 2700, height = 1020, res = 300, type = "cairo")
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
poly_x <- c(edges[1], freq_table$marca_clase, edges[k + 1])
poly_y <- c(0, freq_table$fi, 0)
plot(poly_x, poly_y, type = "n", main = "Polígono de frecuencias (R)",
     xlab = "Consumo (kWh/mes)", ylab = "Frecuencia absoluta (clientes)")
grid()
lines(poly_x, poly_y, type = "o", pch = 19, col = "#2b8cbe")
plot(c(edges[1], freq_table$lim_superior), c(0, freq_table$Hi_pct),
     type = "n", ylim = c(0, 105), main = "Ojiva (R)",
     xlab = "Consumo (kWh/mes)", ylab = "Frecuencia acumulada (%)")
grid()
lines(c(edges[1], freq_table$lim_superior), c(0, freq_table$Hi_pct),
      type = "o", pch = 19, col = "#2b8cbe")
abline(h = 50, col = "#d95f02", lty = 2)
legend("bottomright", cex = 0.75, col = "#d95f02", lty = 2,
       legend = sprintf("50 %% -> Mediana ~ %.0f kWh", median(x)))
dev.off()

#' Barras de frecuencia absoluta por sector, ordenadas por magnitud.
sector_freq <- sort(table(df$sector))
png(file.path(figures_dir, "bar_freq_by_sector.png"),
    width = 1950, height = 960, res = 300, type = "cairo")
par(mar = c(4, 4, 3, 1))
bar_pos <- barplot(sector_freq, col = NA, border = NA,
                   ylim = c(0, max(sector_freq) * 1.2),
                   main = "Frecuencia de clientes por sector (R)",
                   xlab = "Sector", ylab = "Frecuencia absoluta (clientes)")
grid(nx = NA, ny = NULL)
barplot(sector_freq, col = c("#a6bddb", "#74a9cf", "#2b8cbe"),
        add = TRUE, axes = FALSE)
text(x = bar_pos, y = sector_freq + max(sector_freq) * 0.07,
     labels = sprintf("%d (%.0f%%)", sector_freq, sector_freq / n * 100),
     cex = 0.8)
dev.off()

#' Orden de sectores por escala de consumo (no alfabetico) para que las
#' figuras siguientes coincidan con las de Python.
sector_order <- c("Residencial", "Comercial", "Industrial")
df$sector <- factor(df$sector, levels = sector_order)

#' Barras agrupadas media vs mediana por sector, en dos pasadas.
#'
#' La cercania de ambas barras dentro de cada grupo evidencia simetria local
#' pese a la asimetria global de la distribucion.
central_matrix <- rbind(
  Media = tapply(x, df$sector, mean),
  Mediana = tapply(x, df$sector, median)
)
png(file.path(figures_dir, "bar_mean_median_by_sector.png"),
    width = 1950, height = 1080, res = 300, type = "cairo")
par(mar = c(4, 4, 3, 1))
bar_pos <- barplot(central_matrix, beside = TRUE, col = NA, border = NA,
                   ylim = c(0, max(central_matrix) * 1.2),
                   main = "Media y mediana del consumo por sector (R)",
                   xlab = "Sector", ylab = "Consumo (kWh/mes)")
grid(nx = NA, ny = NULL)
barplot(central_matrix, beside = TRUE, col = c("#2b8cbe", "#a6bddb"),
        add = TRUE, axes = FALSE)
text(x = bar_pos, y = central_matrix + max(central_matrix) * 0.04,
     labels = sprintf("%.0f", central_matrix), cex = 0.7)
legend("topleft", cex = 0.75, bty = "n", fill = c("#2b8cbe", "#a6bddb"),
       legend = c("Media", "Mediana"))
dev.off()

#' Diagrama de caja con la media superpuesta, en dos pasadas.
png(file.path(figures_dir, "boxplot_dispersion_by_sector.png"),
    width = 1950, height = 1140, res = 300, type = "cairo")
par(mar = c(4, 4, 3, 1))
boxplot(consumo_kwh ~ sector, data = df, border = NA,
        ylim = c(0, max(x) * 1.12),
        main = "Dispersión del consumo por sector (R)",
        xlab = "Sector", ylab = "Consumo (kWh/mes)")
grid(nx = NA, ny = NULL)
boxplot(consumo_kwh ~ sector, data = df,
        col = c("#a6bddb", "#74a9cf", "#2b8cbe"), add = TRUE, axes = FALSE)
group_means <- tapply(x, df$sector, mean)
group_sd <- tapply(x, df$sector, sd)
points(seq_along(sector_order), group_means, col = "#d95f02", pch = 19)
group_max <- tapply(x, df$sector, max)
text(seq_along(sector_order), group_max + max(x) * 0.05,
     labels = sprintf("sigma = %.0f", group_sd), cex = 0.75)
dev.off()

cat("OK - verificacion cruzada y figuras de R generadas\n")
