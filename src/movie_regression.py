from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


MANUAL_MOVIE_TARGET = "rating_real"
SIMPLE_FEATURE = "reseñas_positivas_pct"
MULTIPLE_FEATURES = [
    "reseñas_positivas_pct",
    "popularidad",
    "presupuesto_millones",
    "experiencia_director",
]


def small_manual_movie_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pelicula": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
                "Pelicula D",
                "Pelicula E",
                "Pelicula F",
                "Pelicula G",
                "Pelicula H",
                "Pelicula I",
                "Pelicula J",
            ],
            "reseñas_positivas_pct": [42, 50, 58, 63, 67, 72, 78, 84, 90, 96],
            "rating_real": [5.1, 5.7, 6.1, 6.4, 6.8, 7.1, 7.5, 8.0, 8.4, 9.0],
            "popularidad": [45, 52, 60, 62, 70, 75, 81, 85, 91, 96],
            "presupuesto_millones": [20, 35, 40, 55, 65, 80, 95, 110, 140, 180],
        }
    )


def generate_large_movie_dataset(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    reviews = rng.uniform(15, 98, n_rows)
    popularity = np.clip(reviews * 0.55 + rng.normal(24, 13, n_rows), 0, 100)
    budget = rng.lognormal(mean=3.2, sigma=0.8, size=n_rows).clip(1, 300)
    experience = rng.gamma(shape=2.8, scale=4.2, size=n_rows).clip(0, 40)
    rating = (
        2.65
        + 0.046 * reviews
        + 0.013 * popularity
        - 0.0018 * budget
        + 0.035 * experience
        + rng.normal(0, 0.42, n_rows)
    )
    return pd.DataFrame(
        {
            "reseñas_positivas_pct": reviews.round(2),
            "popularidad": popularity.round(2),
            "presupuesto_millones": budget.round(2),
            "experiencia_director": experience.round(2),
            "rating_real": np.clip(rating, 1.0, 10.0).round(2),
        }
    )


def predict_simple(X, m: float, b: float) -> np.ndarray:
    return np.asarray(X, dtype=float) * float(m) + float(b)


def compute_errors(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    return pd.DataFrame(
        {
            "rating_predicho": y_pred,
            "error": error,
            "error_absoluto": np.abs(error),
            "error_cuadrado": error**2,
        }
    )


def compute_mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_gradients_simple(X, y_true, y_pred) -> tuple[float, float]:
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    dm = float(np.mean(2 * error * X))
    db = float(np.mean(2 * error))
    return dm, db


def update_simple_params(m: float, b: float, dm: float, db: float, learning_rate: float) -> tuple[float, float]:
    return float(m - learning_rate * dm), float(b - learning_rate * db)


def simple_metrics(y_true, y_pred) -> dict:
    errors = compute_errors(y_true, y_pred)
    mae = float(errors["error_absoluto"].mean())
    mse = float(errors["error_cuadrado"].mean())
    return {"mae": mae, "mse": mse, "rmse": float(np.sqrt(mse))}


def train_simple_steps(
    X,
    y_true,
    m: float,
    b: float,
    learning_rate: float,
    steps: int,
    history: list[dict] | None = None,
) -> tuple[float, float, list[dict], bool]:
    history = list(history or [])
    stable = True
    for _ in range(int(steps)):
        y_pred = predict_simple(X, m, b)
        mse_before = compute_mse(y_true, y_pred)
        dm, db = compute_gradients_simple(X, y_true, y_pred)
        next_m, next_b = update_simple_params(m, b, dm, db, learning_rate)
        next_pred = predict_simple(X, next_m, next_b)
        mse_after = compute_mse(y_true, next_pred)
        if not np.isfinite([next_m, next_b, mse_after]).all() or mse_after > 1e8:
            stable = False
            break
        m, b = next_m, next_b
        history.append(
            {
                "iteracion": len(history) + 1,
                "m": m,
                "b": b,
                "dm": dm,
                "db": db,
                "mse": mse_after,
                "rmse": float(np.sqrt(mse_after)),
                "mse_antes": mse_before,
            }
        )
    return float(m), float(b), history, stable


def normalize_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    values = df[features].astype(float)
    means = values.mean()
    stds = values.std(ddof=0).replace(0, 1)
    normalized = (values - means) / stds
    stats = pd.DataFrame(
        {
            "Variable": features,
            "media antes": means.values,
            "desviacion antes": stds.values,
            "media despues": normalized.mean().values,
            "desviacion despues": normalized.std(ddof=0).values,
        }
    )
    return normalized, means, stds, stats


def predict_multiple(X, weights, bias: float) -> np.ndarray:
    return np.asarray(X, dtype=float) @ np.asarray(weights, dtype=float) + float(bias)


def compute_gradients_multiple(X, y_true, y_pred) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    dw = np.mean(2 * error[:, None] * X, axis=0)
    db = float(np.mean(2 * error))
    return dw, db


def update_multiple_params(weights, bias: float, dw, db: float, learning_rate: float) -> tuple[np.ndarray, float]:
    next_weights = np.asarray(weights, dtype=float) - float(learning_rate) * np.asarray(dw, dtype=float)
    next_bias = float(bias - learning_rate * db)
    return next_weights, next_bias


def train_multiple_steps(
    X,
    y_true,
    weights,
    bias: float,
    learning_rate: float,
    steps: int,
    history: list[dict] | None = None,
) -> tuple[np.ndarray, float, list[dict], bool]:
    history = list(history or [])
    weights = np.asarray(weights, dtype=float)
    stable = True
    for _ in range(int(steps)):
        y_pred = predict_multiple(X, weights, bias)
        dw, db = compute_gradients_multiple(X, y_true, y_pred)
        next_weights, next_bias = update_multiple_params(weights, bias, dw, db, learning_rate)
        next_pred = predict_multiple(X, next_weights, next_bias)
        mse = compute_mse(y_true, next_pred)
        if not np.isfinite(next_weights).all() or not np.isfinite([next_bias, mse]).all() or mse > 1e8:
            stable = False
            break
        weights, bias = next_weights, float(next_bias)
        history.append(
            {
                "iteracion": len(history) + 1,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "bias": bias,
                **{f"w_{idx + 1}": weight for idx, weight in enumerate(weights)},
            }
        )
    return weights, float(bias), history, stable


def sklearn_comparison(df: pd.DataFrame, manual_pred) -> pd.DataFrame:
    X = df[MULTIPLE_FEATURES].astype(float)
    y = df[MANUAL_MOVIE_TARGET].astype(float)
    X_norm, _, _, _ = normalize_features(df, MULTIPLE_FEATURES)
    model = LinearRegression()
    model.fit(X_norm, y)
    sklearn_pred = model.predict(X_norm)
    rows = []
    for name, pred in [("Modelo hecho a mano", manual_pred), ("Modelo scikit-learn", sklearn_pred)]:
        mse = compute_mse(y, pred)
        rows.append(
            {
                "Modelo": name,
                "MAE": mean_absolute_error(y, pred),
                "RMSE": float(np.sqrt(mse)),
                "R2": r2_score(y, pred),
            }
        )
    return pd.DataFrame(rows)
