import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
import lightgbm as lgb


# --- Características del Dataset de California Housing ---

# MedInc: Ingreso Mediano del bloque de censo en decenas de miles de dólares.
#         Es la variable más predictiva. Por ejemplo, un valor de 5.0 representa $50,000.

# HouseAge: Edad Mediana de la vivienda en un bloque de censo.
#           Representa qué tan antiguas son las casas en esa área.

# AveRooms: Número promedio de habitaciones por vivienda.
#           Se calcula dividiendo el total de habitaciones por el número de viviendas en el bloque.

# AveBedrms: Número promedio de dormitorios por vivienda.
#            Se calcula de la misma manera que AveRooms.

# Population: Población total del bloque de censo.
#             Es el número de personas que viven en esa área geográfica específica.

# AveOccup: Número promedio de ocupantes por vivienda.
#           Se calcula como la población del bloque dividida por el número de hogares.

# Latitude: La latitud del bloque de censo.
#           Indica la posición geográfica de norte a sur. (recomentable usar cercanos a 30)

# Longitude: La longitud del bloque de censo.
#            Indica la posición geográfica de este a oeste. (recomentable usar cercanos a -120)


# 1. Cargar el dataset y dividir los datos
print("Cargando el California Housing dataset...")
housing = fetch_california_housing()
data = pd.DataFrame(data=housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target

X = data.drop('MedHouseVal', axis=1)  # Características
y = data['MedHouseVal']  # Variable objetivo

# Dividimos los datos para entrenar cada modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("¡Datos cargados y divididos correctamente!")
print("---")

# 2. Entrenar los 6 modelos
print("Entrenando los 8 modelos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos Lineales
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

lasso_model = Lasso(alpha=0.01, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

# Modelos de Ensemble
forest_reg_model = RandomForestRegressor(random_state=42)
forest_reg_model.fit(X_train, y_train)

gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)

# 1. Validación Cruzada
from sklearn.model_selection import cross_val_score

print("Evaluando Random Forest con validación cruzada...")
scores = cross_val_score(forest_reg_model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Puntuaciones de RMSE por pliegue: {np.round(rmse_scores, 4)}")
print(f"RMSE promedio: {rmse_scores.mean():.4f}")
print(f"Desviación estándar de RMSE: {rmse_scores.std():.4f}")
print("---")

# Regresión de Vectores de Soporte (SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

print("¡Entrenamiento completado para los 8 modelos!")
print("---")

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
}
grid_search = GridSearchCV(forest_reg_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

print("Iniciando la búsqueda de los mejores hiperparámetros...")
grid_search.fit(X, y)

print("¡Búsqueda completada!")
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
best_rmse = np.sqrt(-grid_search.best_score_)
print(f"Mejor RMSE obtenido: {best_rmse:.4f}")

best_forest_model = grid_search.best_estimator_
print("---")

print("Entrenando modelo XGBoost...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

print("Entrenando modelo LightGBM...")
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

print("---")

# 3. Pedir al usuario los datos de la casa
print("Por favor, introduce los siguientes datos de la casa en California:")
user_input = []
feature_names = housing.feature_names
for feature in feature_names:
    value = float(input(f"Introduce el valor para '{feature}': "))
    user_input.append(value)

# Convertir la entrada del usuario a un DataFrame
input_df = pd.DataFrame([user_input], columns=feature_names)
input_df_scaled = scaler.transform(input_df)

# 4. Make and display predictions
def get_prediction_in_dollars(model, input_data):
    predicted_value = model.predict(input_data)[0]
    return predicted_value * 100000

print("---")
print("Prediction Results:")

# Prediction for each model
price_linear = get_prediction_in_dollars(linear_reg_model, input_df)
print(f"Prediction (Linear Regression): ${price_linear:,.2f}")

price_ridge = get_prediction_in_dollars(ridge_model, input_df_scaled)
print(f"Prediction (Ridge Regression): ${price_ridge:,.2f}")

price_lasso = get_prediction_in_dollars(lasso_model, input_df_scaled)
print(f"Prediction (Lasso Regression): ${price_lasso:,.2f}")

price_forest = get_prediction_in_dollars(forest_reg_model, input_df)
print(f"Prediction (Random Forest): ${price_forest:,.2f}")

price_gbr = get_prediction_in_dollars(gbr_model, input_df)
print(f"Prediction (Gradient Boosting): ${price_gbr:,.2f}")

price_svr = get_prediction_in_dollars(svr_model, input_df_scaled)
print(f"Prediction (SVR): ${price_svr:,.2f}")

# --- Añadir las predicciones para XGBoost y LightGBM aquí ---
price_xgb = get_prediction_in_dollars(xgb_model, input_df)
print(f"Predicción (XGBoost): ${price_xgb:,.2f}")

price_lgb = get_prediction_in_dollars(lgb_model, input_df)
print(f"Predicción (LightGBM): ${price_lgb:,.2f}")

print("---")
# 5. Evaluación cuantitativa de los modelos
print("Evaluación del rendimiento en el conjunto de prueba (Test Set):")

models = {
    'LinearRegression': (linear_reg_model, X_test, y_test),
    'Ridge': (ridge_model, X_test_scaled, y_test),
    'Lasso': (lasso_model, X_test_scaled, y_test),
    'RandomForest': (forest_reg_model, X_test, y_test),
    'GradientBoosting': (gbr_model, X_test, y_test),
    'SVR': (svr_model, X_test_scaled, y_test),
    'XGBoost': (xgb_model, X_test, y_test),
    'LightGBM': (lgb_model, X_test, y_test)
}

results = []

for name, (model, X_data, y_true) in models.items():
    y_pred = model.predict(X_data)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])
print("---")
print("Tabla de Métricas de Evaluación:")
print(results_df.to_string())
print("---")

# 6. Análisis de los coeficientes de los modelos lineales
print("Análisis de Coeficientes de Modelos Lineales y Regularizados:")

linear_coefficients = pd.DataFrame(linear_reg_model.coef_, index=housing.feature_names, columns=['Linear Regression'])
ridge_coefficients = pd.DataFrame(ridge_model.coef_, index=housing.feature_names, columns=['Ridge'])
lasso_coefficients = pd.DataFrame(lasso_model.coef_, index=housing.feature_names, columns=['Lasso'])

coefficients_df = pd.concat([linear_coefficients, ridge_coefficients, lasso_coefficients], axis=1)
print(coefficients_df.to_string())
print("---")

print("Análisis de Componentes en Otros Modelos:")

# Usaremos Random Forest como ejemplo, ya que es uno de los mejores
importances = forest_reg_model.feature_importances_
feature_names = housing.feature_names
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Ordenar las características por importancia de forma descendente
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print(feature_importance_df.to_string(index=False))
print("---")

# --- Añadir después de la sección de evaluación de los modelos ---
import matplotlib.pyplot as plt

# Usaremos las predicciones del modelo de Random Forest como ejemplo
y_pred = forest_reg_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores Reales (cien mil USD)")
plt.ylabel("Predicciones del Modelo (cien mil USD)")
plt.title("Valores Reales vs. Predicciones (Random Forest)")
plt.grid(True)
plt.show()

print("Gráfico de Predicciones Generado.")
print("---")

# --- Añadir después de la línea de importaciones ---
import seaborn as sns

# --- Añadir después de cargar los datos del dataset ---
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de Calor de Correlación de las Características")
plt.show()

print("Mapa de Calor de Correlación Generado.")
print("---")