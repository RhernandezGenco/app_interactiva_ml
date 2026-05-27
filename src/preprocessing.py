from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, clean_dataset


def generate_synthetic_dataset(n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    paises = ["Guatemala", "El Salvador", "Honduras", "Costa Rica", "Panama"]
    canales = ["Retail", "Online", "B2B", "Marketplace"]
    segmentos = ["Minorista", "Mayorista", "Corporativo", "Nuevo"]
    categorias = ["Tecnologia", "Hogar", "Oficina", "Moda", "Deportes"]
    subcategorias = ["Accesorios", "Electrodomesticos", "Herramientas", "Ropa", "Muebles"]
    marcas = ["Nova", "Andes", "CasaPlus", "OfficeLab", "Ninja"]
    proveedores = ["Proveedor A", "Proveedor B", "Proveedor C", "Proveedor D"]
    vendedores = ["Ana Lopez", "Maria Garcia", "Carlos Ruiz", "Luis Perez", "Sofia Mendez"]
    pagos = ["Tarjeta", "Efectivo", "Transferencia", "Credito 30 dias", "PayPal"]
    promos = ["Sin promo", "Cupon", "Black Friday", "Liquidacion"]
    prioridades = ["Normal", "Express", "Urgente"]

    cantidad = rng.integers(1, 25, n_rows)
    precio_unitario = rng.lognormal(mean=6.1, sigma=0.65, size=n_rows).round(2)
    descuento = rng.choice([0, 0.05, 0.1, 0.15, 0.25, 0.35], n_rows, p=[0.35, 0.2, 0.18, 0.12, 0.1, 0.05])
    costo_unitario = (precio_unitario * rng.uniform(0.45, 0.85, n_rows)).round(2)
    tipo_cambio = rng.normal(7.8, 0.25, n_rows).round(4)
    venta_neta = (cantidad * precio_unitario * (1 - descuento) * tipo_cambio).round(2)
    margen_gtq = (cantidad * (precio_unitario - costo_unitario) * (1 - descuento) * tipo_cambio).round(2)
    dias_entrega = rng.poisson(4, n_rows) + rng.choice([0, 4, 8], n_rows, p=[0.78, 0.17, 0.05])
    entrega_tardia = (dias_entrega > 7).astype(int)
    stock_disponible = rng.integers(0, 350, n_rows)
    dias_reposicion = rng.integers(2, 30, n_rows)
    rating_cliente = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.08, 0.12, 0.22, 0.32, 0.26])

    canal = rng.choice(canales, n_rows, p=[0.34, 0.36, 0.18, 0.12])
    categoria = rng.choice(categorias, n_rows)
    promocion = rng.choice(promos, n_rows, p=[0.45, 0.25, 0.15, 0.15])
    prioridad = rng.choice(prioridades, n_rows, p=[0.65, 0.25, 0.1])

    risk = (
        -2.4
        + 0.7 * entrega_tardia
        + 1.15 * (descuento >= 0.25)
        + 0.5 * (rating_cliente <= 2)
        + 0.45 * (canal == "Online")
        + 0.4 * (categoria == "Tecnologia")
        + 0.35 * (prioridad == "Urgente")
        + 0.25 * (stock_disponible < 20)
        - 0.000015 * margen_gtq
    )
    probability = 1 / (1 + np.exp(-risk))
    devuelto = rng.binomial(1, np.clip(probability, 0.03, 0.85))

    base_date = pd.Timestamp("2025-01-01")
    dates = base_date + pd.to_timedelta(rng.integers(0, 365, n_rows), unit="D")
    df = pd.DataFrame(
        {
            "id_linea": [f"LIN-{i:06d}" for i in range(1, n_rows + 1)],
            "id_orden": [f"ORD-{i // 2:06d}" for i in range(1, n_rows + 1)],
            "fecha_orden": dates.astype(str),
            "pais": rng.choice(paises, n_rows),
            "canal": canal,
            "segmento_cliente": rng.choice(segmentos, n_rows),
            "rating_cliente": rating_cliente,
            "categoria": categoria,
            "subcategoria": rng.choice(subcategorias, n_rows),
            "marca": rng.choice(marcas, n_rows),
            "proveedor": rng.choice(proveedores, n_rows),
            "vendedor": rng.choice(vendedores, n_rows),
            "metodo_pago": rng.choice(pagos, n_rows),
            "promocion": promocion,
            "prioridad_envio": prioridad,
            "cantidad": cantidad,
            "precio_unitario": precio_unitario,
            "descuento": descuento,
            "costo_unitario": costo_unitario,
            "tipo_cambio": tipo_cambio,
            "venta_neta": venta_neta,
            "margen_gtq": margen_gtq,
            "dias_entrega": dias_entrega,
            "entrega_tardia": entrega_tardia,
            "stock_disponible": stock_disponible,
            "dias_reposicion": dias_reposicion,
            TARGET: devuelto,
        }
    )
    return clean_dataset(df)


def split_feature_types(df: pd.DataFrame, selected_features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [col for col in selected_features if col in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in selected_features if col in CATEGORICAL_FEATURES and col not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    use_scaling: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaling:
        numeric_steps.append(("scaler", StandardScaler()))

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("encoder", encoder),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []
