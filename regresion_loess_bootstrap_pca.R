# Cargar librerías (todas necesarias)
library(GPArotation)# Rotaciones factoriales
library(tidyr)      # Manipulación de datos
# Cargar librerías
library(MVN); library(psych); library(FactoMineR)
library(boot)         # Bootstrap
library(factoextra)   # Visualización de PCA y Clustering
library(ggplot2)      # Gráficos 2D
library(cluster)      # Clustering K-Medoides
library(plotly)       # Gráficos 3D interactivos

datos <- datosEjemplo
datos <- as.data.frame(scale(datos)) # Cargar paquetes necesarios
if (!require(car)) install.packages("car")
library(car)

# Ajustar el modelo de regresión lineal
modelo <- lm(Var1 ~ ., data = datos, na.action = na.omit)

# Tabla ANOVA
cat("=== Tabla ANOVA ===\n")
print(anova(modelo))

# Bondad de ajuste (R² y R² ajustado)
cat("\n=== Bondad de ajuste ===\n")
bondad <- data.frame(
  R2 = summary(modelo)$r.squared,
  R2_ajustado = summary(modelo)$adj.r.squared
)
print(bondad)

# Coeficientes, Tolerancia y VIF
cat("\n=== Tabla de coeficientes, Tolerancia y VIF ===\n")
vif_valores <- vif(modelo)
tolerancia <- 1 / vif_valores
tabla_coef <- summary(modelo)$coefficients
tabla_completa <- cbind(tabla_coef, Tolerancia = tolerancia, VIF = vif_valores)
print(round(tabla_completa, 4))


# Instalar paquetes si es necesario
if (!require(moments)) install.packages("moments")
library(moments)

# Variable dependiente
y <- datos$Var1

# Histograma con curva normal superpuesta
hist(y, breaks = 15, freq = FALSE, col = "lightblue", main = "Histograma de la variable dependiente",
     xlab = "Var1")
curve(dnorm(x, mean = mean(y, na.rm = TRUE), sd = sd(y, na.rm = TRUE)), 
      col = "red", lwd = 2, add = TRUE)

# Gráfico Q-Q
qqnorm(y, main = "Q-Q Plot de la variable dependiente")
qqline(y, col = "red", lwd = 2)

# Test de Shapiro-Wilk
cat("=== Test de Shapiro-Wilk ===\n")
shapiro_result <- shapiro.test(y)
print(shapiro_result)

# Ajustar un modelo lineal para revisar los residuos
modelo_prueba <- lm(Var1 ~ ., data = datos)

# Gráfico de residuos vs valores ajustados
plot(modelo_prueba$fitted.values, modelo_prueba$residuals,
     xlab = "Valores ajustados", ylab = "Residuos",
     main = "Residuos vs Ajustados")
abline(h = 0, col = "red")

library(lmtest)
bptest(modelo_prueba)

library(car)
vif(modelo_prueba)

cooks_d <- cooks.distance(modelo)

# Visualizar con un gráfico
plot(cooks_d, type = "h", main = "Distancia de Cook", ylab = "Distancia", xlab = "Observación")
abline(h = 4/length(cooks_d), col = "red", lty = 2)  # Umbral típico

library(MASS)
modelo_full <- lm(y ~ ., data = datos)
step_model <- stepAIC(modelo_full, direction = "both")
summary(step_model)
vif(step_model)

# Cargar librerías
library(randomForest)
library(ggplot2)
library(caret)

# Cargar datos y excluir outliers (filas 37, 81-84)
datos <- datosEjemplo[-c(37, 81:84), ]

# Escalar variables (opcional, pero útil para comparar coeficientes)
datos_escalados <- as.data.frame(scale(datos))

modelo <- lm(Var1 ~ Var2 + Var3 + Var4 + Var5, data = datos)

bptest(modelo)
vif(modelo)

set.seed(123)  # Para reproducibilidad

# Definir función bootstrap para obtener coeficientes
boot_fn <- function(data, indices) {
  # Crear muestra bootstrap
  datos_boot <- data[indices, ]
  
  # Ajustar el modelo de regresión con las variables seleccionadas
  modelo_boot <- lm(Var1 ~ Var2 + Var3 + Var4 + Var5, data = datos_boot)
  
  # Devolver los coeficientes
  return(coef(modelo_boot))
}

# Ejecutar el bootstrap con 1000 replicaciones
set.seed(123)  # Para reproducibilidad
resultados_boot <- boot(data = datos, statistic = boot_fn, R = 1000)

# Ver los coeficientes promedios y errores estándar
print(resultados_boot)

# Intervalos de confianza percentiles
boot.ci(resultados_boot, type = "perc", index = 2)  # Para el segundo coeficiente (por ejemplo)
boot.ci(resultados_boot, type = "perc", index = 3)  # Para el tercer coeficiente
boot.ci(resultados_boot, type = "perc", index = 4)  # Para el cuarto coeficiente
boot.ci(resultados_boot, type = "perc", index = 5)  # Para el quinto coeficiente

grafico_regresion <- function(variable, datos) {
  
  # Ajuste lineal
  modelo_lineal <- lm(Var1 ~ get(variable), data = datos)
  
  # Ajuste no lineal (LOESS)
  modelo_loess <- loess(Var1 ~ get(variable), data = datos)
  
  # Crear datos de predicción para la regresión lineal
  pred_lineal <- predict(modelo_lineal, newdata = datos)
  
  # Crear datos de predicción para la regresión no lineal (LOESS)
  pred_loess <- predict(modelo_loess, newdata = datos)
  
  # Crear gráfico
  ggplot(datos, aes_string(x = variable, y = "Var1")) +
    geom_point(color = "blue") +  # Puntos de datos
    geom_line(aes(x = datos[[variable]], y = pred_lineal), color = "red", size = 1) +  # Recta de regresión lineal
    geom_line(aes(x = datos[[variable]], y = pred_loess), color = "green", size = 1, linetype = "dashed") +  # Curva LOESS
    labs(title = paste("Regresión para", variable),
         x = variable,
         y = "Satisfacción") +
    theme_minimal() +
    theme(legend.position = "none")
}

# Lista de variables independientes
variables <- c("Var2", "Var3",  "Var5",  "Var4")

# Generar gráficos para cada variable
for (var in variables) {
  print(grafico_regresion(var, datos))
}

# Cargar datos y preparación
data(datosEjemplo)  # Asegurarse de cargar el dataset correcto
datos <- scale(datosEjemplo)  # Estandarización

# Aplicar PCA sin asumir normalidad
pca <- prcomp(datos_scaled, center = TRUE, scale. = TRUE)

# Ver los resultados del PCA
summary(pca)
pca_result <- prcomp(datos, center = TRUE, scale. = TRUE)
# Graficar el scree plot (proporción de varianza explicada)
par(mar = c(4, 4, 2, 1))  # Márgenes más estrechos (abajo, izq, arriba, der)
plot(pca, main = "Scree Plot")

# Graficar los primeros dos componentes principales
biplot(pca_result, main = "Biplot del PCA")

# Obtener la matriz de loadings
loadings_matrix <- pca_result$rotation

# Mostrar la matriz de loadings
print(loadings_matrix)

if (!require("GPArotation")) install.packages("GPArotation")
library(GPArotation)

# Aplicar rotaciones
varimax_rot <- varimax(loadings_matrix)
quartimax_rot <- quartimax(loadings_matrix)
equamax_rot <- equamax(loadings_matrix)

# Obtener matrices rotadas
varimax_loadings <- varimax_rot$loadings
quartimax_loadings <- round(quartimax_rot$loadings)
equamax_loadings <- round(equamax_rot$loadings)

# 3. Mostrar resultados
print("Varimax:")
print(varimax_loadings)

print("Quartimax:")
print(quartimax_loadings)

print("Equamax:")
print(equamax_loadings)
# --------------------------------------
# PASO 2: Seleccionar solo los 3 primeros CP
# --------------------------------------
loadings_matrix <- pca_result$rotation[, 1:3]  # CP1, CP2, CP3

# --------------------------------------
# PASO 3: Aplicar rotaciones (con normalización Kaiser)
# --------------------------------------
varimax_rot <- varimax(loadings_matrix, normalize = TRUE)
quartimax_rot <- quartimax(loadings_matrix, normalize = TRUE)
equamax_rot <- equamax(loadings_matrix, normalize = TRUE)

# --------------------------------------
# PASO 4: Mostrar resultados (solo loadings > |0.3|)
# --------------------------------------
print("=== VARIMAX (CP1-CP3) ===")
print(varimax_rot$loadings)

print("=== QUARTIMAX (CP1-CP3) ===")
print(quartimax_rot$loadings)

print("=== EQUAMAX (CP1-CP3) ===")
print(equamax_rot$loadings)

if (!require("tidyr")) install.packages("tidyr")
library(tidyr)

prepare_plot_data <- function(loadings, method) {
  df <- as.data.frame(unclass(loadings))
  df$Variable <- rownames(df)
  df_long <- pivot_longer(df, 
                          cols = -Variable, 
                          names_to = "Componente", 
                          values_to = "Loading")
  df_long$Method <- method
  return(df_long)
}

# Combinar todos los datos (sin rotar + rotaciones)
plot_data <- rbind(
  prepare_plot_data(loadings_matrix, "Sin rotar"),  # <-- Añadido
  prepare_plot_data(varimax_rot$loadings, "Varimax"),
  prepare_plot_data(quartimax_rot$loadings, "Quartimax"),
  prepare_plot_data(equamax_rot$loadings, "Equamax")
)

# --------------------------------------
# PASO 5: Graficar comparación (4 métodos)
# --------------------------------------
ggplot(plot_data, aes(x = Componente, y = Variable, fill = Loading)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(Loading, 2)), size = 3, color = "black") +
  scale_fill_gradient2(
    low = "#2E86C1", 
    mid = "white", 
    high = "#CB4335", 
    midpoint = 0,
    limits = c(-1, 1)  # Fijar escala para comparación justa
  ) +
  facet_wrap(~ Method, ncol = 2) +  # 2 columnas para mejor visualización
  theme_minimal(base_size = 12) +
  labs(
    title = "Comparación de Loadings: Componentes Originales vs Rotados",
    subtitle = "Primeros 3 Componentes Principales (CP1-CP3)",
    x = "Componente Principal",
    y = "Variables",
    fill = "Valor de Loading"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 11)
  )


# Gráfico (Biplot mejorado)
ggplot(var_data, aes(x = CP1, y = CP2)) +
  # Círculo de correlación (radio = 1)
  geom_path(
    data = circleFun(center = c(0, 0), diameter = 2, npoints = 100),
    aes(x = x, y = y), 
    color = "gray80", 
    linetype = "dashed"
  ) +
  # Vectores de variables (flechas)
  geom_segment(
    aes(x = 0, y = 0, xend = CP1, yend = CP2),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "#E15759",
    linewidth = 0.7
  ) +
  # Etiquetas de variables
  geom_text(
    aes(label = Variable, 
        x = CP1 * 1.1,  # Ajuste para evitar solapamiento
        y = CP2 * 1.1),
    color = "#4E79A7",
    size = 4,
    fontface = "bold"
  ) +
  # Escalas y temas
  scale_x_continuous(limits = c(-1.2, 1.2)) +
  scale_y_continuous(limits = c(-1.2, 1.2)) +
  labs(
    title = "Variables en el Plano CP1-CP2",
    x = paste0("CP1 (", round(pca_result$eig[1, 2], 1), "% var.)"),
    y = paste0("CP2 (", round(pca_result$eig[2, 2], 1), "% var.)")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    panel.grid.major = element_line(color = "gray90")
  )

# Función auxiliar para dibujar el círculo de correlación
circleFun <- function(center = c(0, 0), diameter = 1, npoints = 100) {
  r <- diameter / 2
  tt <- seq(0, 2 * pi, length.out = npoints)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

# Realizar PCA (si no lo has hecho aún)
pca_result <- PCA(datos, graph = FALSE)

# Extraer la matriz de puntuaciones (scores) de las dos primeras PCs
scores_pc1_pc2 <- as.data.frame(pca_result$ind$coord[, 1:2])
colnames(scores_pc1_pc2) <- c("PC1", "PC2")

# Mostrar primeras filas
head(scores_pc1_pc2, 10)


library(cluster)    # Para silhouette y pam
library(factoextra) # Visualización
library(NbClust)    # Métodos avanzados
library(ggdendro)
library(patchwork) # Para organizar múltiples gráficos

# Datos: Coordenadas de las 3 primeras PCs
pca_final <- PCA(datos, scale.unit = TRUE, graph = FALSE)

# Tomamos las coordenadas de las 3 primeras PC
pca_data <- as.data.frame(pca_final$ind$coord[,1:3])
pca_data <- as.data.frame(pca_final$ind$coord[, 1:3])
dist_matrix <- dist(pca_data, method = "euclidean")

# Métodos de linkage a comparar
methods <- c("ward.D2", "complete", "average", "single", "centroid")

# Función para generar y plotear dendrogramas
plot_dendro <- function(method) {
  hc <- hclust(dist_matrix, method = method)
  dend <- as.dendrogram(hc)
  
  # Calcular número óptimo de clusters (silhouette)
  sil_width <- sapply(2:10, function(k) {
    mean(silhouette(cutree(hc, k), dist_matrix)[, "sil_width"])
  })
  optimal_k <- which.max(sil_width) + 1
  
  # Colorear dendrograma
  dend <- color_branches(dend, k = optimal_k)
  
  # Plot
  ggdendrogram(dend, theme_dendro = FALSE) +
    labs(
      title = paste("Linkage:", method),
      subtitle = paste("Clusters óptimos (Silhouette):", optimal_k),
      x = "Muestras",
      y = "Altura"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
}

# Generar todos los dendrogramas
dend_plots <- lapply(methods, plot_dendro)

# Organizar en una sola figura
wrap_plots(dend_plots, ncol = 2) + 
  plot_annotation(title = "Comparación de Métodos de Linkage",
                  theme = theme(plot.title = element_text(size = 16, face = "bold")))



# Datos: Coordenadas de las 3 primeras PCs (tras PCA + bootstrap)
pca_data <- as.data.frame(pca_final$ind$coord[, 1:3])

# --- Método 1: Silhouette ---
fviz_nbclust(pca_data, FUN = pam, method = "silhouette", k.max = 30) +
  labs(title = "Método de la Silueta") +
  theme_minimal()

# --- Método 2: Elbow (WSS) ---
fviz_nbclust(pca_data, FUN = pam, method = "wss", k.max = 15) +  # Ejemplo para k=3
  labs(title = "Método del Codo") +
  theme_minimal()

k_optimo <- 2  # Valor determinado
kmedoids_final <- pam(pca_data, k = k_optimo)

# Asignar clusters a los datos
pca_data$Cluster <- as.factor(kmedoids_final$clustering)

# Visualización 2D
fviz_cluster(kmedoids_final, data = pca_data, 
             palette = "Set1", 
             ellipse.type = "convex",
             ggtheme = theme_minimal(),
             main = "Clusters K-Medoides (k=3) en PC1-PC2")
plot_ly(pca_data, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3, 
        color = ~Cluster, colors = c("#FF0000", "#00FF00", "#0000FF"),
        marker = list(size = 5)) %>%
  layout(scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")),
         title = "Clusters en 3 PCs (k=3)")



kmedoids_result <- pam(pca_data, k = 2)

(resultados_clusters <- data.frame(
  Observación = rownames(pca_data),
  PC1 = pca_data$Dim.1,
  PC2 = pca_data$Dim.2, 
  PC3 = pca_data$Dim.3,
  Cluster = as.factor(kmedoids_final$clustering)
))


# Crear tabla combinada
(tabla_clusters <- data.frame(
  Observación = rownames(datos),
  PC1 = pca_data$Dim.1,
  PC2 = pca_data$Dim.2,
  PC3 = pca_data$Dim.3,
  Cluster = as.factor(kmedoids_result$clustering)
))



library(MVN)
library(psych)
library(FactoMineR)
library(boot)
library(factoextra)
library(ggplot2)
library(cluster)
library(plotly)
library(RColorBrewer)
library(fpc)

kmeans_avanzado <- function(data, k, n_inicial = 25, iter_max = 100, 
                            grafico_2d = TRUE, grafico_3d = TRUE) {
  # Validación de datos
  if (!all(sapply(data, is.numeric))) {
    stop("Todos los datos deben ser numéricos")
  }
  
  # Escalado opcional (recomendado para K-Means)
  datos_escalados <- scale(data)
  
  # Aplicar K-Means con múltiples inicializaciones
  set.seed(123)
  modelo_kmeans <- kmeans(datos_escalados, centers = k, nstart = n_inicial, 
                          iter.max = iter_max)
  
  # Resultados detallados
  resultados <- list(
    clusters = modelo_kmeans$cluster,
    centros = modelo_kmeans$centers,
    totss = modelo_kmeans$totss,
    withinss = modelo_kmeans$withinss,
    tot.withinss = modelo_kmeans$tot.withinss,
    betweenss = modelo_kmeans$betweenss,
    size = modelo_kmeans$size
  )
  
  # Métricas de calidad
  distancia <- dist(datos_escalados)
  resultados$silhouette <- mean(silhouette(modelo_kmeans$cluster, distancia)[, "sil_width"])
  resultados$calinski <- calinhara(datos_escalados, modelo_kmeans$cluster)
  
  # Visualización si se solicita
  if (grafico_2d || grafico_3d) {
    datos_plot <- as.data.frame(datos_escalados)
    datos_plot$Cluster <- as.factor(modelo_kmeans$cluster)
    
    if (grafico_2d && ncol(data) >= 2) {
      resultados$plot_2d <- ggplot(datos_plot, aes(x = .data[[names(data)[1]]], 
                                                   y = .data[[names(data)[2]]], 
                                                   color = Cluster)) +
        geom_point(size = 3, alpha = 0.7) +
        stat_ellipse(level = 0.95) +
        scale_color_brewer(palette = "Set1") +
        labs(title = paste("K-Means con k =", k),
             x = names(data)[1],
             y = names(data)[2]) +
        theme_minimal()
    }
    
    if (grafico_3d && ncol(data) >= 3) {
      resultados$plot_3d <- plot_ly(datos_plot, 
                                    x = ~.data[[names(data)[1]]],
                                    y = ~.data[[names(data)[2]]],
                                    z = ~.data[[names(data)[3]]],
                                    color = ~Cluster,
                                    colors = RColorBrewer::brewer.pal(k, "Set1"),
                                    marker = list(size = 5)) %>%
        layout(scene = list(xaxis = list(title = names(data)[1]),
                            yaxis = list(title = names(data)[2]),
                            zaxis = list(title = names(data)[3])),
               title = paste("K-Means 3D (k =", k, ")"))
    }
  }
  
  return(resultados)
}

# Usando solo las 2 primeras componentes principales
pca_data <- as.data.frame(pca_orig$ind$coord[, 1:3])
colnames(pca_data) <- c("PC1", "PC2", "PC3")

# Análisis con k=2
resultados_k3 <- kmeans_avanzado(pca_data, k = 2)
print(resultados_k3[c("silhouette", "calinski")])
resultados_k3$plot_2d
resultados_k3$plot_3d

# Función para comparar múltiples valores de k
comparar_k <- function(data, k_values = 2:25) {
  resultados <- lapply(k_values, function(k) {
    res <- kmeans_avanzado(data, k, grafico_2d = FALSE, grafico_3d = FALSE)
    data.frame(k = k,
               silhouette = res$silhouette,
               calinski = res$calinski,
               tot.withinss = res$tot.withinss)
  })
  
  do.call(rbind, resultados)
}

# Ejecutar comparación
comparacion <- comparar_k(pca_data)

# Visualizar comparación
ggplot(comparacion, aes(x = k, y = silhouette)) +
  geom_line(color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "Análisis de Silueta para diferentes k",
       x = "Número de clusters (k)",
       y = "Ancho medio de silueta") +
  theme_minimal()

ggplot(comparacion, aes(x = k, y = tot.withinss)) +
  geom_line(color = "firebrick") +
  geom_point(size = 3, color = "firebrick") +
  labs(title = "Índice de Calinski-Harabasz para diferentes k",
       x = "Número de clusters (k)",
       y = "Índice CH") +
  theme_minimal()

# Función para encontrar el k óptimo
encontrar_k_optimo <- function(data, max_k = 25) {
  comparacion <- comparar_k(data, 2:max_k)
  
  # Silhouette máximo
  k_silhouette <- comparacion$k[which.max(comparacion$silhouette)]
  
  # Calinski máximo
  k_calinski <- comparacion$k[which.max(comparacion$calinski)]
  
  # Método del codo (mayor reducción en WSS)
  diffs <- diff(comparacion$tot.withinss)
  k_elbow <- comparacion$k[which.min(diffs) + 1]
  
  list(k_silhouette = k_silhouette,
       k_calinski = k_calinski,
       k_elbow = k_elbow,
       comparacion = comparacion)
}

# Ejecutar optimización
k_optimo <- encontrar_k_optimo(pca_data)
print(paste("k óptimo por silueta:", k_optimo$k_silhouette))
print(paste("k óptimo por Calinski-Harabasz:", k_optimo$k_calinski))
print(paste("k óptimo por método del codo:", k_optimo$k_elbow))

# Ejecutar con el mejor k según silueta
mejor_k <- k_optimo$k_silhouette
resultados_finales <- kmeans_avanzado(pca_data, k = 3)

# Mostrar resultados
resultados_finales$plot_2d +
  labs(title = paste("Mejor solución K-Means (k =", mejor_k, ")"))

resultados_finales$plot_3d

# Mostrar asignación de clusters
data.frame(pca_data, Cluster = resultados_finales$clusters)

