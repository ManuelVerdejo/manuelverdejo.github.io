"""
Enhanced time series forecasting pipeline
- Mejoras en generación de datos, reproducibilidad
- Ingeniería de características extendida
- Mejores prácticas para Prophet, XGBoost y modelo híbrido
- Guardado de modelos y visualizaciones, métricas y diagnóstico
- Plots adicionales: descomposición, ACF/PACF, residuales, importancia de variables, heatmap estacional

Uso: python forecasting_pipeline_enhanced.py
Opciones: revisar variables al inicio del script para modificar rutas/fechas.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Prefer matplotlib para gráficas; seaborn solo para estilos ligeros si se necesita
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)

# Modelado
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Diagnóstico y descomposición
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Intenta importar SHAP (opcional)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ========== CONFIG ===========
RNG_SEED = 42
np.random.seed(RNG_SEED)

START_DATE = '2022-01-01'
END_DATE = '2024-12-31'
SPLIT_DATE = '2024-07-01'  # inclusive for test
MODEL_DIR = Path('models')
FIG_DIR = Path('figures')
REPORTS_DIR = Path('reports')
for p in (MODEL_DIR, FIG_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Prophet defaults
PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
}

# XGBoost default search space (Randomized)
XGB_PARAM_DIST = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('forecasting')

# ========== 0. UTILIDADES ===========

def set_seed(seed: int = 42):
    np.random.seed(seed)


def ensure_datetime(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def save_fig(fig, fname: str):
    path = FIG_DIR / fname
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info(f"Figura guardada en: {path}")

# ========== 1. DATOS Y PREPROCESADO ===========

def create_advanced_dummy_data(start_date: str, end_date: str, seed: int = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'date': dates})
    n = len(df)

    # Componentes
    trend = np.linspace(10, 300, n)  # tendencia lineal suave
    weekly = 20 * np.sin(2 * np.pi * (df['date'].dt.dayofweek) / 7)  # semanal
    yearly = 30 * np.cos(2 * np.pi * (df['date'].dt.dayofyear) / 365.25)  # anual
    noise = np.random.normal(0, 8, n)

    # Marketing spend con ciclos mensuales
    marketing_spend = 12 + 8 * np.sin(2 * np.pi * np.arange(n) / 30.5) + np.random.normal(0, 2, n)
    marketing_spend = np.clip(marketing_spend, 0, None)

    # Festivos simples
    is_december_holiday = (df['date'].dt.month == 12) & (df['date'].dt.day >= 20)
    holiday_effect = is_december_holiday.astype(int) * 60

    # Promociones aleatorias
    promo_days = np.random.choice(n, size=int(0.02 * n), replace=False)
    promo = np.zeros(n)
    promo[promo_days] = np.random.uniform(30, 80, len(promo_days))

    # Combinar todo
    sales = trend + weekly + yearly + 1.5 * marketing_spend + holiday_effect + promo + noise
    df['sales'] = np.maximum(0, sales).round().astype(int)
    df['marketing_spend'] = marketing_spend.round(2)

    # Outliers multiplicativos
    outlier_idx = np.random.choice(n, size=max(1, int(0.01 * n)), replace=False)
    df.loc[outlier_idx, 'sales'] = (df.loc[outlier_idx, 'sales'] * np.random.uniform(2, 4, len(outlier_idx))).astype(int)

    return df



def handle_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = df[(df[column] < lower) | (df[column] > upper)].shape[0]
    logger.info(f"Detectados {n_outliers} outliers en '{column}' (IQR). Se recortarán a límites).")
    df[column] = np.clip(df[column], lower, upper)
    return df

# ========== 2. INGENIERÍA DE CARACTERÍSTICAS ===========

def create_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    df = ensure_datetime(df, date_col).copy()
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    # Cycliс features (sin/cos) para capturar periodicidad sin saltos
    df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['sin_month'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['cos_month'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df


def create_lag_rolling_features(df: pd.DataFrame, target: str, lags=None, windows=None) -> pd.DataFrame:
    df = df.copy().set_index('date')
    if lags is None:
        lags = [7, 14, 28]
    if windows is None:
        windows = [7, 14]
    for lag in lags:
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    for w in windows:
        df[f'{target}_roll_mean_{w}'] = df[target].shift(1).rolling(window=w).mean()
        df[f'{target}_roll_std_{w}'] = df[target].shift(1).rolling(window=w).std()
    df = df.dropna()
    return df.reset_index()

# ========== 3. MODELOS ===========

class ProphetModel:
    def __init__(self, params=None, holidays=None):
        params = params or {}
        self.model = Prophet(**{**PROPHET_PARAMS, **params}, holidays=holidays)
        self.fitted = False

    def add_regressor(self, name, **kwargs):
        self.model.add_regressor(name, **kwargs)

    def fit(self, df: pd.DataFrame):
        """df debe tener columnas 'ds' y 'y' y opcionalmente regresores"""
        logger.info("Entrenando Prophet...")
        self.model.fit(df)
        self.fitted = True
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError('Prophet no está entrenado')
        return self.model.predict(future_df)


class XGBoostModel:
    def __init__(self, target_col: str = 'sales'):
        self.target_col = target_col
        self.model = None
        self.features = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_datetime(df, 'date').copy()
        df = create_time_features(df)
        # mantener regresores conocidos
        # si existe marketing_spend, se conservará automáticamente
        df = create_lag_rolling_features(df, self.target_col)
        # Asegurarse que no queden columnas no numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # forzar target incluido
        if self.target_col not in numeric_cols:
            raise ValueError(f"Target '{self.target_col}' no está presente en columnas numéricas")
        return df

    def fit(self, df_train: pd.DataFrame, n_iter: int = 20):
        logger.info("Entrenando XGBoost con RandomizedSearchCV (TimeSeriesSplit)...")
        df_train_prepared = self._prepare(df_train)
        X = df_train_prepared.drop(columns=[self.target_col, 'date'])
        y = df_train_prepared[self.target_col]
        self.features = X.columns.tolist()

        tscv = TimeSeriesSplit(n_splits=3)
        xgb_est = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=RNG_SEED)

        search = RandomizedSearchCV(
            estimator=xgb_est,
            param_distributions=XGB_PARAM_DIST,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            verbose=1,
            random_state=RNG_SEED
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        logger.info(f"Mejores hiperparámetros XGBoost: {search.best_params_}")
        return self

    def predict(self, df_test: pd.DataFrame, df_train_history: pd.DataFrame) -> pd.DataFrame:
        # Para crear lags correctamente concatenamos historico + test
        df_combined = pd.concat([df_train_history, df_test], ignore_index=True)
        df_prepared = self._prepare(df_combined)
        # Tomar solamente las filas correspondientes al test
        test_mask = df_prepared['date'] >= pd.to_datetime(SPLIT_DATE)
        X_test = df_prepared.loc[test_mask, self.features]
        preds = self.model.predict(X_test)
        results = df_prepared.loc[test_mask, ['date']].copy()
        results['prediction'] = preds
        return results.set_index('date')

    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"XGBoost guardado en: {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        logger.info(f"XGBoost cargado desde: {path}")
        return self


class HybridProphetXGB:
    """Prophet predice tendencia+estacionalidad; XGBoost modela residuos."""
    def __init__(self):
        self.prophet = ProphetModel()
        self.xgb = XGBoostModel(target_col='residual')
        self.prophet_train_forecast = None

    def fit(self, df_train: pd.DataFrame):
        logger.info('Entrenando modelo híbrido (Prophet + XGBoost sobre residuos)')
        # Preparar datos para Prophet
        df_p = df_train.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y', 'marketing_spend']].copy()
        # Registrar regresores si existen
        if 'marketing_spend' in df_p.columns:
            self.prophet.add_regressor('marketing_spend')
        self.prophet.fit(df_p)

        # Forecast sobre entrenamiento para obtener residuos
        forecast_train = self.prophet.predict(df_p[['ds', 'marketing_spend']].rename(columns={'ds': 'ds'}))
        df_train_res = df_train.copy().reset_index(drop=True)
        df_train_res['residual'] = df_train_res['sales'].values - forecast_train['yhat'].values
        self.prophet_train_forecast = forecast_train

        # Entrenar XGBoost sobre residuos
        # Nota: XGBoostModel espera columna 'date' y target 'residual'
        self.xgb.fit(df_train_res[['date', 'residual']], n_iter=10)
        return self

    def predict(self, df_test: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
        # 1) Prophet forecast
        df_test_p = df_test.rename(columns={'date': 'ds'})[['ds', 'marketing_spend']].copy()
        prophet_forecast = self.prophet.predict(df_test_p.rename(columns={'ds': 'ds'}))

        # 2) XGBoost residuos
        # Construir df de entrenamiento de residuos (date + residual)
        df_train_res = df_train.copy().reset_index(drop=True)
        df_train_res['residual'] = df_train_res['sales'].values - self.prophet_train_forecast['yhat'].values

        # Construimos df_test_for_xgb con columna residual placeholder
        df_test_for_xgb = df_test.copy()
        df_test_for_xgb['residual'] = 0.0

        xgb_res_preds = self.xgb.predict(df_test_for_xgb[['date', 'residual']], df_train_res[['date', 'residual']])

        # 3) Combinar
        final = prophet_forecast[['ds', 'yhat']].copy()
        final.index = pd.to_datetime(final['ds'])
        xgb_res_preds = xgb_res_preds.reindex(index=final.index)

        final['residual_pred'] = xgb_res_preds['prediction'].values
        final['prediction'] = np.maximum(0, final['yhat'].values + final['residual_pred'].values)
        results = final[['prediction']].copy()
        results.index.name = 'date'
        return results

# ========== 4. EVALUACIÓN Y GRÁFICOS ===========

def metrics_table(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.align(y_pred, join='inner')
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = (np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))).mean() * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape}


def plot_comparison(y_true: pd.Series, predictions: dict, title: str = 'Comparativa de modelos'):
    fig, ax = plt.subplots()
    ax.plot(y_true.index, y_true.values, label='Real', linewidth=2, color='black')
    for name, pred in predictions.items():
        ax.plot(pred.index, pred.values, label=name)
    ax.set_title(title)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas')
    ax.legend()
    save_fig(fig, 'comparison.png')
    plt.show()


def plot_decomposition(series: pd.Series, model: str = 'additive'):
    res = seasonal_decompose(series, model=model, period=365)
    fig = res.plot()
    save_fig(fig, 'decomposition.png')
    plt.show()


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, name: str):
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(res)
    axes[0].set_title(f'Residuales {name} (tiempo)')
    axes[1].hist(res, bins=30)
    axes[1].set_title('Histograma residuales')
    plot_acf(res.dropna(), ax=axes[2], lags=40)
    axes[2].set_title('ACF residuales')
    save_fig(fig, f'residuals_{name}.png')
    plt.show()


def plot_feature_importance(xgb_model: xgb.XGBRegressor, feature_names, top_n: int = 20):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        xgb.plot_importance(xgb_model, max_num_features=top_n, ax=ax)
        save_fig(fig, 'xgb_feature_importance.png')
        plt.show()
    except Exception as e:
        logger.warning(f"No se pudo generar feature importance: {e}")

# ========== 5. UTILIDADES AVANZADAS: INTERVALOS Y SHAP (OPCIONAL) ===========

def bootstrap_prediction_intervals(residuals: np.ndarray, point_preds: np.ndarray, alpha: float = 0.05, n_boot: int = 1000):
    """Crea intervalos de predicción agregando muestras bootstrap de residuos al pronóstico puntual."""
    rng = np.random.RandomState(RNG_SEED)
    sims = np.zeros((n_boot, len(point_preds)))
    for i in range(n_boot):
        sampled = rng.choice(residuals, size=len(point_preds), replace=True)
        sims[i] = point_preds + sampled
    lower = np.percentile(sims, 100 * alpha / 2, axis=0)
    upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return lower, upper

# ========== 6. EJECUCIÓN PRINCIPAL ===========

def main(verbose: bool = True):
    set_seed(RNG_SEED)

    # 1. Generar datos
    df = create_advanced_dummy_data(START_DATE, END_DATE, seed=RNG_SEED)
    df = handle_outliers_iqr(df, 'sales')

    train_df = df[df['date'] < pd.to_datetime(SPLIT_DATE)].reset_index(drop=True)
    test_df = df[df['date'] >= pd.to_datetime(SPLIT_DATE)].reset_index(drop=True)
    y_true = test_df.set_index('date')['sales']

    # 2. Prophet
    pmodel = ProphetModel()
    # Añadir marketing_spend como regresor si existe
    if 'marketing_spend' in train_df.columns:
        pmodel.add_regressor('marketing_spend')
    pmodel.fit(train_df.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y', 'marketing_spend']])
    future = test_df[['date', 'marketing_spend']].rename(columns={'date': 'ds'})
    prophet_forecast = pmodel.predict(future)
    prophet_preds = prophet_forecast.set_index('ds')['yhat']

    # 3. XGBoost
    xmodel = XGBoostModel(target_col='sales')
    xmodel.fit(train_df, n_iter=10)
    xgb_preds = xmodel.predict(test_df[['date', 'sales', 'marketing_spend']], train_df[['date', 'sales', 'marketing_spend']])

    # 4. Híbrido
    hybrid = HybridProphetXGB()
    hybrid.fit(train_df)
    hybrid_preds = hybrid.predict(test_df[['date', 'sales', 'marketing_spend']], train_df[['date', 'sales', 'marketing_spend']])

    # 5. Evaluación
    df_metrics = []
    m_prophet = metrics_table(y_true, prophet_preds.rename_axis('date'))
    m_xgb = metrics_table(y_true, xgb_preds['prediction'])
    m_hybrid = metrics_table(y_true, hybrid_preds['prediction'])
    df_metrics.append({'model': 'Prophet', **m_prophet})
    df_metrics.append({'model': 'XGBoost', **m_xgb})
    df_metrics.append({'model': 'Hybrid', **m_hybrid})
    metrics_df = pd.DataFrame(df_metrics).set_index('model')
    metrics_df.to_csv(REPORTS_DIR / 'model_metrics.csv')
    logger.info('\n' + metrics_df.to_string())

    # 6. Plots
    plot_comparison(y_true, {
        'Prophet': prophet_preds.rename_axis('date'),
        'XGBoost': xgb_preds['prediction'],
        'Hybrid': hybrid_preds['prediction']
    })

    # Descomposición de la serie objetivo
    plot_decomposition(df.set_index('date')['sales'])

    # Residuales
    plot_residuals(y_true, prophet_preds.rename_axis('date'), 'Prophet')
    plot_residuals(y_true, xgb_preds['prediction'], 'XGBoost')
    plot_residuals(y_true, hybrid_preds['prediction'], 'Hybrid')

    # Importancia de features XGBoost (si disponible)
    if isinstance(xmodel.model, xgb.XGBRegressor):
        plot_feature_importance(xmodel.model, xmodel.features)

    # SHAP (opcional)
    if SHAP_AVAILABLE and isinstance(xmodel.model, xgb.XGBRegressor):
        logger.info('Generando explicaciones SHAP (resumen)')
        # reconstruit X_test used for predictions
        df_comb = pd.concat([train_df, test_df], ignore_index=True)
        df_prep = xmodel._prepare(df_comb)
        test_mask = df_prep['date'] >= pd.to_datetime(SPLIT_DATE)
        X_test = df_prep.loc[test_mask, xmodel.features]
        explainer = shap.Explainer(xmodel.model)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=True)

    # Guardar modelos
    joblib.dump(pmodel.model, MODEL_DIR / 'prophet_model.joblib')
    xmodel.save(MODEL_DIR / 'xgb_model.joblib')
    joblib.dump(hybrid, MODEL_DIR / 'hybrid_model.joblib')
    logger.info('Modelos guardados.')

    # Guardar predicciones
    preds_df = pd.DataFrame({
        'y_true': y_true,
        'prophet': prophet_preds.values,
        'xgb': xgb_preds['prediction'].values,
        'hybrid': hybrid_preds['prediction'].values
    }, index=y_true.index)
    preds_df.to_csv(REPORTS_DIR / 'predictions.csv')
    logger.info('Predicciones guardadas.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline forecasting avanzado')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(verbose=args.verbose)
