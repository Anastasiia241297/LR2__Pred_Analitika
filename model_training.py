import os
import matplotlib
import pickle
matplotlib.use('Agg')
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Абсолютные пути к директориям
BASE_DIR = r'C:\Users\hhh\pythonProject12'
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')

# Добавляем путь для сохранения моделей
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)  # Создаем папку если ее нет

def save_model(model, filename):
    """Сохраняет модель в папку models в формате pickle"""
    try:
        full_path = os.path.join(MODELS_DIR, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'Модель сохранена: {full_path}')
        return full_path
    except Exception as e:
        print(f'Ошибка сохранения модели: {e}')
        return None

# Создаем директории для графиков и метрик, если они не существуют
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(fig, filename):
    """Сохраняет график в папку plots с абсолютным путем"""
    try:
        full_path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(full_path)
        plt.close(fig)
        print(f'График сохранен: {full_path}')
        return full_path
    except Exception as e:
        print(f'Ошибка сохранения графика: {e}')
        return None


def save_json(data, filename):
    """Сохраняет JSON в папку metrics с абсолютным путем"""
    try:
        full_path = os.path.join(METRICS_DIR, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Данные сохранены: {full_path}')
        return full_path
    except Exception as e:
        print(f'Ошибка сохранения JSON: {e}')
        return None


def compare_split_methods(X, y):
    """Сравниваем разные методы разделения данных"""
    methods = {
        'TimeSeriesSplit': TimeSeriesSplit(n_splits=5),
        'RandomSplit': KFold(n_splits=5, shuffle=True, random_state=42),
        'BlockingSplit': TimeSeriesSplit(n_splits=5, gap=24)
    }

    results = {}
    for name, splitter in methods.items():
        cv_scores = []
        for train_idx, val_idx in splitter.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.03,
                depth=7,
                loss_function='MAE',
                verbose=0
            )
            model.fit(X_train_scaled, y_train)
            cv_scores.append(mean_absolute_error(y_val, model.predict(X_val_scaled)))

        results[name] = np.mean(cv_scores)
        print(f'{name} MAE: {results[name]:.3f}')

    try:
        fig = plt.figure(figsize=(10, 5))
        plt.bar(results.keys(), results.values())
        plt.title('Сравнение методов разделения данных')
        plt.ylabel('MAE')
        plot_path = save_plot(fig, 'split_methods_comparison.png')
    except Exception as e:
        print(f"Ошибка при построении графика: {e}")
        plot_path = None

    return min(results, key=results.get), plot_path


def train_and_evaluate_model(df_for_model):
    # Улучшенная предобработка данных
    df = df_for_model.dropna().copy()

    cols_to_drop = [col for col in ['Unnamed: 0', 'index'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    target = 'raw_mix.lab.measure.sito_009'
    features = [col for col in df.columns if col != target]

    # Feature engineering
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_part'] = (df['hour'] // 6).astype(int)
        features.extend(['hour', 'day_part'])

    # Добавляем признаки
    df['target_lag1'] = df[target].shift(1)
    df['target_lag2'] = df[target].shift(2)
    df['rolling_range_24h'] = df[target].rolling(24).max() - df[target].rolling(24).min()
    df['was_high'] = (df[target] > df[target].quantile(0.9)).astype(int)
    df['was_low'] = (df[target] < df[target].quantile(0.1)).astype(int)

    for col in ['raw_mix.consumption_mean_15_len_10_shift', 'temp.out_mean_25_len_5_shift']:
        if col in features:
            df[f'{col}_rolling_std_5'] = df[col].rolling(5).std()
            df[f'{col}_rolling_mean_3'] = df[col].rolling(3).mean()
            features.extend([f'{col}_rolling_std_5', f'{col}_rolling_mean_3'])

    if 'raw_mix.consumption_mean_15_len_10_shift' in features and 'temp.out_mean_25_len_5_shift' in features:
        df['consumption_temp_ratio'] = df['raw_mix.consumption_mean_15_len_10_shift'] / (
                    df['temp.out_mean_25_len_5_shift'] + 1e-6)
        features.append('consumption_temp_ratio')

    features.extend(['target_lag1', 'target_lag2', 'rolling_range_24h', 'was_high', 'was_low'])
    features = [col for col in df.columns if col != target]
    df = df.dropna()

    # Разделение данных
    print("\n=== Сравнение методов разделения ===")
    X = df[features]
    y = df[target]
    best_method, split_comparison_plot = compare_split_methods(X, y)
    print(f"Лучший метод: {best_method}")

    test_size = 0.2
    if best_method == 'RandomSplit':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Скалирование
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Модель CatBoost
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=42,
        od_type='Iter',
        od_wait=100,
        l2_leaf_reg=5,
        grow_policy='Lossguide',
        verbose=0,
        bootstrap_type='Bayesian',
        score_function='L2'
    )

    # Кросс-валидация
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=0
        )
        cv_scores.append(mean_absolute_error(y_val_fold, model.predict(X_val_fold)))

    print(f'Cross-validation MAE scores: {cv_scores}')
    print(f'Mean CV MAE: {np.mean(cv_scores):.3f}')

    # Финальное обучение
    model.fit(
        X_train_scaled, y_train,
        eval_set=(X_test_scaled, y_test),
        use_best_model=True,
        verbose=0
    )

    # Оценка
    y_pred = model.predict(X_test_scaled)

    # Метрики
    extreme_mask = (y_test < y_test.quantile(0.1)) | (y_test > y_test.quantile(0.9))
    mae_extreme = mean_absolute_error(y_test[extreme_mask], y_pred[extreme_mask])

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mae_baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
    mae_baseline_shift = mean_absolute_error(y_test[:-1], y_test[1:])

    y_test_diff = y_test[1:].values - y_test[:-1].values
    y_pred_diff = y_pred[1:] - y_pred[:-1]
    same_direction = np.sum((y_test_diff * y_pred_diff) > 0) / len(y_test_diff)

    print(f'\nДоля сонаправленных изменений: {same_direction:.3f}')
    print(f'MAE на экстремальных значениях: {mae_extreme:.3f}')
    print(f'Mean Absolute Error: {mae:.3f}')
    print(f'Baseline MAE (mean): {mae_baseline:.3f}')
    print(f'Baseline MAE (shift): {mae_baseline_shift:.3f}')

    # Визуализация
    predictions_plot_path = None
    try:
        fig = plt.figure(figsize=(15, 6))
        x_axis = np.arange(len(y_test))
        plt.plot(x_axis, y_test.values, label='Факт', linewidth=2, color='blue')
        plt.plot(x_axis, y_pred, label='Прогноз', linestyle='--', color='orange')
        plt.scatter(x_axis[extreme_mask], y_test[extreme_mask], color='red', label='Экстремальные значения', zorder=5)
        plt.title('Фактические и прогнозируемые значения')
        plt.xlabel('Номер наблюдения (отсортировано по времени)')
        plt.ylabel(target)
        plt.legend()
        plt.grid(True)
        predictions_plot_path = save_plot(fig, 'predictions_plot.png')
    except Exception as e:
        print(f"Ошибка при построении графика предсказаний: {e}")

    # Важность признаков
    feature_importance_path = None
    try:
        feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)[:20]
        fig = plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_imp.values, y=feature_imp.index)
        plt.title('Важность признаков (Top 20)')
        plt.tight_layout()
        feature_importance_path = save_plot(fig, 'feature_importance.png')
    except Exception as e:
        print(f"Ошибка при построении графика важности признаков: {e}")

    # Линейная регрессия
    linear_reg_plot_path = None
    mae_lin = None
    try:
        X_train_lin = X_train.copy().dropna(axis=1)
        X_test_lin = X_test.copy()[X_train_lin.columns]

        scaler_lin = StandardScaler()
        X_train_lin_scaled = scaler_lin.fit_transform(X_train_lin)
        X_test_lin_scaled = scaler_lin.transform(X_test_lin)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train_lin_scaled, y_train)
        y_pred_lin = lin_reg.predict(X_test_lin_scaled)
        mae_lin = mean_absolute_error(y_test, y_pred_lin)
        print(f'\nLinear Regression MAE: {mae_lin:.3f}')

        fig = plt.figure(figsize=(12, 6))
        feature_importances = lin_reg.coef_
        indices = np.argsort(np.abs(feature_importances))[::-1]
        plt.bar(range(len(indices)), feature_importances[indices], align='center')
        plt.xticks(range(len(indices)), np.array(X_train_lin.columns)[indices], rotation=45, ha='right')
        plt.title('Feature Importances (Linear Regression)')
        plt.ylabel('Coefficient Value')
        plt.tight_layout()
        linear_reg_plot_path = save_plot(fig, 'linear_regression_feature_importance.png')
    except Exception as e:
        print(f"Ошибка при работе с линейной регрессией: {e}")

    # Сохранение результатов
    results = {
        'features': features,
        'metrics': {
            'mae': float(mae),
            'mae_extreme': float(mae_extreme),
            'mae_baseline': float(mae_baseline),
            'mae_baseline_shift': float(mae_baseline_shift),
            'same_direction_ratio': float(same_direction),
            'mse': float(mse),
            'cv_scores': [float(score) for score in cv_scores],
            'mean_cv_mae': float(np.mean(cv_scores)),
            'mae_linear_regression': float(mae_lin) if mae_lin is not None else None
        },
        'best_split_method': best_method,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'plots': {
            'split_methods_comparison': split_comparison_plot,
            'predictions_plot': predictions_plot_path,
            'feature_importance': feature_importance_path,
            'linear_regression_feature_importance': linear_reg_plot_path
        }
    }

    save_json(results, f'result_{results["timestamp"]}.json')

    model_path = save_model(model, f'model_{results["timestamp"]}.pkl')
    results['model_path'] = model_path  # Добавляем путь к модели в результаты

    # Обновляем сохранение JSON с новыми данными
    save_json(results, f'result_{results["timestamp"]}.json')

    return model


if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, 'data', 'processed', 'mart.csv')
    df_for_model = pd.read_csv(data_path)
    trained_model = train_and_evaluate_model(df_for_model)