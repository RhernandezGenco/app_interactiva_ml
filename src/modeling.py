from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor, split_feature_types, transformed_feature_names
from src.utils import TARGET


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict
    report: str
    confusion: np.ndarray
    roc: pd.DataFrame | None
    feature_names: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray


def build_estimator(model_name: str, params: dict | None = None, class_weight=None):
    params = params or {}
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1500, C=float(params.get("C", 1.0)), class_weight=class_weight)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 5)))
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(params.get("random_state", 42)),
            class_weight=class_weight,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 120)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(params.get("random_state", 42)),
            class_weight=class_weight,
            n_jobs=-1,
        )
    raise ValueError(f"Modelo no soportado: {model_name}")


def safe_train_test_split(X, y, test_size: float, random_state: int):
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def oversample_minority(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    counts = y_train.value_counts()
    if len(counts) < 2 or counts.min() == counts.max():
        return X_train, y_train

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_needed = counts[majority_class] - counts[minority_class]
    minority_index = y_train[y_train == minority_class].index
    sampled_index = minority_index.to_series().sample(
        n=n_needed,
        replace=True,
        random_state=random_state,
    )
    X_extra = X_train.loc[sampled_index.to_numpy()]
    y_extra = y_train.loc[sampled_index.to_numpy()]
    X_balanced = pd.concat([X_train, X_extra], ignore_index=True)
    y_balanced = pd.concat([y_train.reset_index(drop=True), y_extra.reset_index(drop=True)], ignore_index=True)
    return X_balanced, y_balanced


def train_model(
    df: pd.DataFrame,
    selected_features: list[str],
    model_name: str,
    use_scaling: bool,
    test_size: float,
    random_state: int,
    params: dict | None = None,
    balance_strategy: str = "class_weight",
) -> TrainResult:
    data = df.dropna(subset=[TARGET]).copy()
    X = data[selected_features].copy()
    y = data[TARGET].astype(int)

    if y.nunique() < 2:
        raise ValueError("El target necesita al menos dos clases: 0 y 1.")
    if len(data) < 10:
        raise ValueError("Se necesitan al menos 10 filas para entrenar y probar el modelo.")

    numeric_features, categorical_features = split_feature_types(data, selected_features)
    preprocessor = build_preprocessor(numeric_features, categorical_features, use_scaling)
    estimator_params = dict(params or {})
    estimator_params["random_state"] = random_state
    supports_class_weight = model_name in {"Logistic Regression", "Decision Tree", "Random Forest"}
    class_weight = "balanced" if balance_strategy == "class_weight" and supports_class_weight else None
    estimator = build_estimator(model_name, estimator_params, class_weight=class_weight)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size, random_state)
    train_distribution_before = y_train.value_counts().sort_index().to_dict()
    if balance_strategy == "oversample":
        X_fit, y_fit = oversample_minority(X_train, y_train, random_state)
    else:
        X_fit, y_fit = X_train, y_train
    train_distribution_after = y_fit.value_counts().sort_index().to_dict()

    pipeline.fit(X_fit, y_fit)
    y_pred = pipeline.predict(X_test)

    proba = None
    roc = None
    roc_auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, proba)
            fpr, tpr, thresholds = roc_curve(y_test, proba)
            roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        except Exception:
            roc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
        "test_rows": len(X_test),
        "train_rows": len(X_train),
        "fit_rows": len(X_fit),
        "balance_strategy": balance_strategy,
        "train_distribution_before": train_distribution_before,
        "train_distribution_after": train_distribution_after,
    }
    report = classification_report(y_test, y_pred, zero_division=0)
    confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])
    feature_names = transformed_feature_names(pipeline.named_steps["preprocess"])

    return TrainResult(
        pipeline=pipeline,
        metrics=metrics,
        report=report,
        confusion=confusion,
        roc=roc,
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def compare_redundant_features(
    df_base: pd.DataFrame,
    df_redundant: pd.DataFrame,
    selected_features: list[str],
    model_name: str,
    use_scaling: bool,
    test_size: float,
    random_state: int,
    balance_strategy: str = "class_weight",
) -> pd.DataFrame:
    redundant_cols = [col for col in df_redundant.columns if col.endswith("_dup")]
    rows = []
    for label, data, features in [
        ("Sin variables redundantes", df_base, selected_features),
        ("Con variables redundantes", df_redundant, selected_features + redundant_cols),
    ]:
        result = train_model(data, features, model_name, use_scaling, test_size, random_state, balance_strategy=balance_strategy)
        rows.append({"experimento": label, **{k: v for k, v in result.metrics.items() if k in ["accuracy", "precision", "recall", "f1"]}})
    return pd.DataFrame(rows)


def overfitting_curve(
    df: pd.DataFrame,
    selected_features: list[str],
    use_scaling: bool,
    test_size: float,
    random_state: int,
    min_samples_leaf: int,
    max_depth_limit: int = 20,
    balance_strategy: str = "class_weight",
) -> pd.DataFrame:
    rows = []
    for depth in range(1, max_depth_limit + 1):
        result = train_model(
            df,
            selected_features,
            "Decision Tree",
            use_scaling,
            test_size,
            random_state,
            {"max_depth": depth, "min_samples_leaf": min_samples_leaf},
            balance_strategy=balance_strategy,
        )
        rows.append(
            {
                "max_depth": depth,
                "accuracy_train": result.metrics["train_accuracy"],
                "accuracy_test": result.metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def compute_feature_importance(result: TrainResult, random_state: int = 42) -> pd.DataFrame:
    model = result.pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_") and result.feature_names:
        values = model.feature_importances_
        names = result.feature_names
    else:
        transformed_test = result.pipeline.named_steps["preprocess"].transform(result.X_test)
        importance = permutation_importance(
            model,
            transformed_test,
            result.y_test,
            n_repeats=8,
            random_state=random_state,
            scoring="f1",
        )
        values = importance.importances_mean
        names = result.feature_names or [f"feature_{i}" for i in range(len(values))]

    return (
        pd.DataFrame({"variable": names, "importancia": values})
        .sort_values("importancia", ascending=False)
        .head(10)
    )
