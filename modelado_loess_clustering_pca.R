# Cargar librerías
library(MVN); library(psych); library(FactoMineR)
library(boot)         # Bootstrap
library(factoextra)   # Visualización de PCA y Clustering
library(ggplot2)      # Gráficos 2D
library(cluster)      # Clustering K-Medoides
library(plotly)       # Gráficos 3D interactivos

datos <- datos_cliente
datos <- datos[-56, ]

# Cargar paquetes necesarios
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


# Crear un nuevo conjunto de datos sin la observación 56
datos <- datos[-56, ]

# Ajustar modelo LOESS multivariado con 3 predictores
modelo_loess <- loess(Var1 ~ Var2 +
                        Var4 + Var5,
                      data = datos, span = 1)

# Ajustar modelo LOESS multivariado con 3 predictores
modelo_loess2 <- loess(Var1 ~ Var2 +
                        Var4 + Var5 + Var3,
                      data = datos, span = 1)

# Ver resumen del modelo
summary(modelo_loess)

# Predicciones del modelo LOESS
predicciones <- predict(modelo_loess)
predicciones2 <- predict(modelo_loess2)
# Calcular el pseudo-R² (1 - SSE/SST)
SSE <- sum((datos$Var1 - predicciones)^2)
SSE2 <- sum((datos$Var1 - predicciones2)^2)
SST <- sum((datos$Var1 - mean(datos$Var1))^2)
pseudo_R2 <- 1 - SSE/SST
pseudo_R22 <- 1 - SSE2/SST
# Error cuadrático medio (MSE)
MSE <- mean((datos$Var1 - predicciones)^2)
MSE2 <- mean((datos$Var1 - predicciones2)^2)
# Mostrar resultados
cat("Pseudo-R²:", round(pseudo_R2, 3), "\n")
cat("Pseudo-R²:", round(pseudo_R22, 3), "\n")
cat("MSE:", round(MSE, 3), "\n")
cat("MSE:", round(MSE2, 3), "\n")

plot(datos$Var2, predicciones, 
     main = "Efecto de Pregunta 2 en la predicción",
     xlab = "Pregunta 2", ylab = "Predicción",
     pch = 19, col = "blue")

# Añadir recta de regresión lineal (interpolación simple)
abline(lm(predicciones ~ datos$Var2), col = "red", lwd = 2)

# Opcional: Añadir línea LOESS (suavizado)
lines(lowess(datos$Var2, predicciones), col = "green", lwd = 2)

# Leyenda
legend("topleft", legend = c("Regresión lineal", "LOESS"), 
       col = c("red", "green"), lwd = 2)

plot(datos$Var4, predicciones, 
     main = "Efecto de Var4 en la predicción",
     xlab = "Var4", ylab = "Predicción",
     pch = 19, col = "orange")

# Recta de regresión lineal
abline(lm(predicciones ~ datos$Var4), col = "purple", lwd = 2)

# Línea LOESS
lines(lowess(datos$Var4, predicciones), col = "darkblue", lwd = 2)

# Leyenda
legend("topleft", legend = c("Regresión lineal", "LOESS"), 
       col = c("purple", "darkblue"), lwd = 2)

plot(datos$Var5, predicciones, 
     main = "Efecto de Var5 en la predicción",
     xlab = "Var5", ylab = "Predicción",
     pch = 19, col = "darkgreen")

# Recta de regresión lineal
abline(lm(predicciones ~ datos$Var5), col = "magenta", lwd = 2)

# Línea LOESS
lines(lowess(datos$Var5, predicciones), col = "black", lwd = 2)

# Leyenda
legend("topleft", legend = c("Regresión lineal", "LOESS"), 
       col = c("magenta", "black"), lwd = 2)

plot(datos$Var5, predicciones, 
     main = "Efecto de Var5 en la predicción",
     xlab = "Var5", ylab = "Predicción")

# --------------------------------------------------------------------
datos <- datos_cliente
datos <- as.data.frame(scale(datos))

# 1. Normalidad multivariante
mvn(datos, mvnTest = "mardia")

# 2. Test de esfericidad de Bartlett
cortest.bartlett(cor(datos), nrow(datos))

# 3. KMO
KMO(datos)$MSA  # >0.5 es aceptable

pca_boot <- function(data, idx) {
  PCA(data[idx, ], graph = FALSE)$eig[,1]  # Autovalores
}
resultados <- boot(datos, pca_boot, R = 1000)

# Estabilidad de los componentes
quantile(resultados$t[,1], c(0.025, 0.975))  # IC 95% para el 1er componente

# PCA original
pca_orig <- PCA(datos, graph = FALSE)
fviz_screeplot(pca_orig, addlabels = TRUE)  

# Distribución bootstrap del 1er componente
hist(resultados$t[,1], main = "Autovalor PC1 (Bootstrap)")

# Cargas bootstrap (primeras 2 PCs)
cargas <- sapply(1:1000, function(i) {
  PCA(datos[sample(nrow(datos), replace = TRUE), ], graph = FALSE)$var$coord[,1:2]
})
apply(cargas, 1, quantile, c(0.025, 0.975))  # IC95% por variable


# Función para extraer % varianza en cada réplica bootstrap
pca_var_boot <- function(data, idx) {
  pca <- PCA(data[idx, ], graph = FALSE)
  return(pca$eig[, 2])  # % varianza por componente
}

# Aplicar bootstrap (ej: 1000 réplicas)
set.seed(123)
boot_results <- boot(datos, pca_var_boot, R = 1000)

# Intervalos de confianza (95%) para cada PC
ic_pc1 <- quantile(boot_results$t[, 1], c(0.025, 0.975))  # PC1
ic_pc2 <- quantile(boot_results$t[, 2], c(0.025, 0.975))  # PC2
ic_pc3 <- quantile(boot_results$t[, 3], c(0.025, 0.975))  # PC2
ic_pc4 <- quantile(boot_results$t[, 4], c(0.025, 0.975))  # PC2

print(paste("PC1: IC95% =", round(ic_pc1[1], 1), "% a", round(ic_pc1[2], 1), "%"))
print(paste("PC2: IC95% =", round(ic_pc2[1], 1), "% a", round(ic_pc2[2], 1), "%"))
print(paste("PC3: IC95% =", round(ic_pc3[1], 1), "% a", round(ic_pc3[2], 1), "%"))
print(paste("PC4: IC95% =", round(ic_pc4[1], 1), "% a", round(ic_pc4[2], 1), "%"))


# Calcular media e IC95% de los autovalores
autovalores_medios <- apply(boot_results$t, 2, mean)
autovalores_li <- apply(boot_results$t, 2, quantile, probs = 0.025)
autovalores_ls <- apply(boot_results$t, 2, quantile, probs = 0.975)

# Crear dataframe para el gráfico
df_autovalores <- data.frame(
  PC = paste0("PC", 1:length(autovalores_medios)),
  Autovalor = autovalores_medios,
  LI = autovalores_li,
  LS = autovalores_ls
)

# Gráfico con ggplot2
ggplot(df_autovalores, aes(x = PC, y = Autovalor)) +
  geom_col(fill = "#4E79A7", alpha = 0.8, width = 0.7) +
  geom_errorbar(aes(ymin = LI, ymax = LS), width = 0.2, color = "#E15759", linewidth = 1) +
  geom_point(size = 3, color = "#F28E2B") +
  labs(
    title = "Autovalores (Varianza) de los PCs con IC95% (Bootstrap)",
    x = "Componente Principal",
    y = "Autovalor (Varianza Explicada)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(size = 10)
  )

# Función para PCA y extracción de cargas en cada réplica bootstrap
get_loadings <- function(data, idx) {
  pca <- PCA(data[idx, ], graph = FALSE)
  return(as.vector(pca$var$coord))  # Aplanar matriz de cargas (loadings)
}
boot_loadings <- boot(datos, get_loadings, R = 1000)
# Organizar resultados en una matriz: variables x PCs
n_vars <- ncol(datos)
n_pcs <- ncol(PCA(datos, graph = FALSE)$var$coord)  # Número de PCs retenidos
loadings_estables <- matrix(apply(boot_loadings$t, 2, mean), nrow = n_vars, byrow = FALSE)
rownames(loadings_estables) <- colnames(datos)
colnames(loadings_estables) <- paste0("PC", 1:n_pcs)

# Mostrar matriz de cargas promedio
print(round(loadings_estables, 3))

library(reshape2)
library(viridis)

# Preparar datos para heatmap
loadings_melted <- melt(loadings_estables)
colnames(loadings_melted) <- c("Variable", "PC", "Loading")

# Heatmap
ggplot(loadings_melted, aes(x = PC, y = Variable, fill = Loading)) +
  geom_tile(color = "white") +
  scale_fill_viridis(option = "magma", direction = -1) +
  labs(
    title = "Heatmap de Cargas Factoriales (Promedio Bootstrap)",
    x = "Componente Principal",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0))
# Realizar PCA (si no lo has hecho ya)
pca_result <- PCA(datos, graph = FALSE)

# Extraer coordenadas de las variables (loadings)
var_coord <- as.data.frame(pca_result$var$coord[, 1:2])
colnames(var_coord) <- c("CP1", "CP2")
var_coord$Variable <- rownames(var_coord)

# Calcular la contribución de cada variable a los CPs
var_contrib <- as.data.frame(pca_result$var$contrib[, 1:2])
colnames(var_contrib) <- c("Contrib_CP1", "Contrib_CP2")

# Combinar coordenadas y contribuciones
var_data <- cbind(var_coord, var_contrib)

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
install.packages("dendextend")  # Si no lo tienes instalado
library(dendextend)             # Cargar el paquete
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

k_optimo <- 3  # Valor determinado
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



k_optimo <- 4  # Valor determinado
kmedoids_final <- pam(pca_data, k = k_optimo)

# Asignar clusters a los datos
pca_data$Cluster <- as.factor(kmedoids_final$clustering)

# Visualización 2D
fviz_cluster(kmedoids_final, data = pca_data, 
             palette = "Set1", 
             ellipse.type = "convex",
             ggtheme = theme_minimal(),
             main = "Clusters K-Medoides (k=4) en PC1-PC2")
plot_ly(pca_data, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3, 
        color = ~Cluster, colors = c("#FF0000", "#00FF00", "#0000FF"),
        marker = list(size = 5)) %>%
  layout(scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")),
         title = "Clusters en 3 PCs (k=4)")



k_optimo <- 10  # Valor determinado
kmedoids_final <- pam(pca_data, k = k_optimo)

# Asignar clusters a los datos
pca_data$Cluster <- as.factor(kmedoids_final$clustering)

# Visualización 2D
fviz_cluster(kmedoids_final, data = pca_data, 
             palette = "Set1", 
             ellipse.type = "convex",
             ggtheme = theme_minimal(),
             main = "Clusters K-Medoides (k=10) en PC1-PC2")
plot_ly(pca_data, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3, 
        color = ~Cluster, colors = c("#FF0000", "#00FF00", "#0000FF"),
        marker = list(size = 5)) %>%
  layout(scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")),
         title = "Clusters en 3 PCs (k=10)")


kmedoids_result <- pam(pca_data, k = 4)

# Crear tabla con los datos originales y asignación de clusters
tabla_clusters <- data.frame(
  Observación = rownames(datos),
  PC1 = pca_data$PC1,
  PC2 = pca_data$PC2,
  PC3 = pca_data$PC3,
  Cluster = kmedoids_result$clustering
)



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

# Usando solo las 3 primeras componentes principales
pca_data <- as.data.frame(pca_orig$ind$coord[, 1:3])
colnames(pca_data) <- c("PC1", "PC2", "PC3")

# Análisis con k=3
resultados_k3 <- kmeans_avanzado(pca_data, k = 3)
print(resultados_k3[c("silhouette", "calinski")])
resultados_k3$plot_2d
resultados_k3$plot_3d

# Análisis con k=4
resultados_k4 <- kmeans_avanzado(pca_data, k = 4)
print(resultados_k4[c("silhouette", "calinski")])
resultados_k4$plot_2d
resultados_k4$plot_3d

# Análisis con k=10
resultados_k10 <- kmeans_avanzado(pca_data, k = 10,)
print(resultados_k10[c("silhouette", "calinski")])
resultados_k10$plot_2d
resultados_k10$plot_3d

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

