from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET = "devuelto"

NUMERIC_FEATURES = [
    "rating_cliente",
    "cantidad",
    "precio_unitario",
    "descuento",
    "costo_unitario",
    "tipo_cambio",
    "venta_neta",
    "margen_gtq",
    "dias_entrega",
    "entrega_tardia",
    "stock_disponible",
    "dias_reposicion",
]

CATEGORICAL_FEATURES = [
    "pais",
    "canal",
    "segmento_cliente",
    "categoria",
    "subcategoria",
    "marca",
    "proveedor",
    "vendedor",
    "metodo_pago",
    "promocion",
    "prioridad_envio",
]

EXPECTED_COLUMNS = [
    "id_linea",
    "id_orden",
    "fecha_orden",
    *CATEGORICAL_FEATURES[:3],
    "rating_cliente",
    *CATEGORICAL_FEATURES[3:],
    "cantidad",
    "precio_unitario",
    "descuento",
    "costo_unitario",
    "tipo_cambio",
    "venta_neta",
    "margen_gtq",
    "dias_entrega",
    "entrega_tardia",
    "stock_disponible",
    "dias_reposicion",
    TARGET,
]

COLUMN_ALIASES = {
    "descuento_pct": "descuento",
    "tipo_cambio_gtq": "tipo_cambio",
    "venta_neta_gtq": "venta_neta",
    "dias_reposicion_proveedor": "dias_reposicion",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    renamed.columns = [str(col).strip() for col in renamed.columns]
    return renamed


def validate_dataset(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if TARGET not in df.columns:
        raise ValueError("El CSV debe incluir la columna target 'devuelto'.")

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        warnings.append(
            "Faltan columnas esperadas. La app seguirá con las columnas disponibles: "
            + ", ".join(missing)
        )

    values = set(pd.Series(df[TARGET]).dropna().unique().tolist())
    if not values.issubset({0, 1, "0", "1", False, True}):
        warnings.append("La columna 'devuelto' se convertirá a 0/1; revisa valores no binarios.")
    return warnings


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_column_names(df.copy())
    validate_dataset(clean)

    for col in NUMERIC_FEATURES:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in clean.columns:
            clean[col] = clean[col].astype("string").fillna("Desconocido").str.strip()
            clean[col] = clean[col].replace({"": "Desconocido"})

    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce").fillna(0).astype(int)
    clean[TARGET] = clean[TARGET].clip(0, 1)
    return clean


def available_features(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def find_leakage_columns(df: pd.DataFrame) -> list[str]:
    suspicious_words = ("devolucion", "devolución", "devuelto", "motivo")
    return [
        col
        for col in df.columns
        if col != TARGET and any(word in col.lower() for word in suspicious_words)
    ]


def add_redundant_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    enriched = df.copy()
    specs = {
        "venta_neta": "venta_neta_dup",
        "margen_gtq": "margen_gtq_dup",
        "precio_unitario": "precio_unitario_dup",
        "cantidad": "cantidad_dup",
    }
    for source, target in specs.items():
        if source in enriched.columns:
            std = pd.to_numeric(enriched[source], errors="coerce").std()
            noise = rng.normal(0, max(float(std) * 0.015, 0.01), len(enriched))
            enriched[target] = pd.to_numeric(enriched[source], errors="coerce") + noise
    return enriched


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["variable_1", "variable_2", "correlacion"])
    corr = numeric.corr().abs()
    rows = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= threshold:
                rows.append({"variable_1": col_a, "variable_2": col_b, "correlacion": value})
    return pd.DataFrame(rows).sort_values("correlacion", ascending=False)
