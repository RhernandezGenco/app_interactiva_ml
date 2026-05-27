from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text

from src.modeling import (
    compare_redundant_features,
    compute_feature_importance,
    overfitting_curve,
    train_model,
)
from src.plots import (
    before_after_scaling,
    confusion_matrix_plot,
    correlation_heatmap,
    feature_importance_plot,
    metrics_comparison,
    missing_values,
    overfitting_line,
    roc_curve_plot,
    split_bar,
    target_distribution,
)
from src.preprocessing import generate_synthetic_dataset
from src.utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_redundant_features,
    available_features,
    clean_dataset,
    find_leakage_columns,
    high_correlation_pairs,
    validate_dataset,
)


st.set_page_config(page_title="Laboratorio ML: Devoluciones", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 8px;
    }
    .concept-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 6px 0 12px 0;
    }
    .section-rule {border-top: 1px solid #e5e7eb; margin: 1.4rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(raw_bytes.decode("utf-8")))


@st.cache_data(show_spinner=False)
def cached_synthetic_data(n_rows: int, random_state: int) -> pd.DataFrame:
    return generate_synthetic_dataset(n_rows=n_rows, random_state=random_state)


@st.cache_data(show_spinner=False)
def cached_train(
    df: pd.DataFrame,
    selected_features: list[str],
    model_name: str,
    use_scaling: bool,
    test_size: float,
    random_state: int,
    params: dict,
    balance_strategy: str,
):
    return train_model(
        df,
        selected_features,
        model_name,
        use_scaling,
        test_size,
        random_state,
        params,
        balance_strategy=balance_strategy,
    )


def section(title: str, caption: str | None = None) -> None:
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader(title)
    if caption:
        st.write(caption)


def metric_row(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("F1-score", f"{metrics['f1']:.3f}")
    auc = metrics.get("roc_auc")
    cols[4].metric("ROC AUC", "N/A" if auc is None else f"{auc:.3f}")


def questions(lines: list[str]) -> None:
    st.info("Preguntas para pensar: " + " · ".join(lines))


def variance_table(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "varianza": values.var(),
                "desviacion_std": values.std(),
                "media": values.mean(),
                "min": values.min(),
                "max": values.max(),
                "% nulos": values.isna().mean() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("varianza", ascending=False)


def scaling_stats(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [col for col in numeric_cols if col in df.columns][:4]
    if not cols:
        return pd.DataFrame(), pd.DataFrame()
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(values)
    scaled = StandardScaler().fit_transform(imputed)
    before = pd.DataFrame({"variable": cols, "media": imputed.mean(axis=0), "std": imputed.std(axis=0)})
    after = pd.DataFrame({"variable": cols, "media": scaled.mean(axis=0), "std": scaled.std(axis=0)})
    return before, after


def balance_label_to_strategy(label: str) -> str:
    mapping = {
        "Sin balanceo": "none",
        "Class weight balanced": "class_weight",
        "Sobremuestreo clase minoritaria": "oversample",
    }
    return mapping[label]


def logistic_equation(result, max_terms: int = 12) -> tuple[str, pd.DataFrame]:
    model = result.pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return "", pd.DataFrame()

    coefficients = pd.DataFrame(
        {
            "variable_transformada": result.feature_names,
            "coeficiente": model.coef_[0],
        }
    )
    coefficients["impacto_abs"] = coefficients["coeficiente"].abs()
    coefficients = coefficients.sort_values("impacto_abs", ascending=False)
    intercept = float(model.intercept_[0])
    terms = [
        f"({row.coeficiente:+.3f} * {row.variable_transformada})"
        for row in coefficients.head(max_terms).itertuples(index=False)
    ]
    equation = f"logit(p) = {intercept:.3f} " + " ".join(terms)
    return equation, coefficients


def decision_tree_rules(result, max_depth: int = 4) -> str:
    model = result.pipeline.named_steps["model"]
    if not hasattr(model, "tree_"):
        return ""
    names = result.feature_names or [f"feature_{idx}" for idx in range(model.n_features_in_)]
    return export_text(model, feature_names=names, max_depth=max_depth)


def default_prediction_values(df_model: pd.DataFrame, features: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    values = {}
    base_features = [col for col in features if not col.endswith("_dup")]
    for feature in base_features:
        if feature in numeric_cols or pd.api.types.is_numeric_dtype(df_model[feature]):
            series = pd.to_numeric(df_model[feature], errors="coerce")
            values[feature] = float(series.median()) if pd.notna(series.median()) else 0.0
        else:
            mode = df_model[feature].astype("string").fillna("Desconocido").mode()
            values[feature] = str(mode.iloc[0]) if not mode.empty else "Desconocido"

    for feature in features:
        if feature.endswith("_dup"):
            source = feature.replace("_dup", "")
            if source in values:
                values[feature] = values[source]
            else:
                series = pd.to_numeric(df_model[feature], errors="coerce")
                values[feature] = float(series.median()) if pd.notna(series.median()) else 0.0

    return pd.DataFrame([{feature: values[feature] for feature in features}])


def prediction_column_config(df_model: pd.DataFrame, features: list[str], numeric_cols: list[str]) -> dict:
    config = {}
    for feature in features:
        if feature in numeric_cols or feature.endswith("_dup") or pd.api.types.is_numeric_dtype(df_model[feature]):
            series = pd.to_numeric(df_model[feature], errors="coerce")
            config[feature] = st.column_config.NumberColumn(
                feature,
                help=f"Rango observado: {series.min():.2f} a {series.max():.2f}",
            )
        else:
            options = (
                df_model[feature]
                .astype("string")
                .fillna("Desconocido")
                .drop_duplicates()
                .sort_values()
                .head(200)
                .tolist()
            )
            config[feature] = st.column_config.SelectboxColumn(feature, options=options or ["Desconocido"])
    return config


with st.sidebar:
    st.title("Datos")
    uploaded = st.file_uploader("Cargar CSV de ventas", type=["csv"])
    synthetic_rows = st.slider("Filas sintéticas", 200, 5000, 1200, 100)
    synthetic_seed = st.number_input("Semilla dataset", value=42, step=1)

try:
    if uploaded is not None:
        df_raw = read_uploaded_csv(uploaded.getvalue())
        df = clean_dataset(df_raw)
        source_label = f"CSV cargado: {uploaded.name}"
    else:
        df = cached_synthetic_data(int(synthetic_rows), int(synthetic_seed))
        source_label = "Dataset sintético generado automáticamente"
    dataset_warnings = validate_dataset(df)
except Exception as exc:
    st.error(f"No se pudo preparar el dataset: {exc}")
    st.stop()

numeric_available = available_features(df, NUMERIC_FEATURES)
categorical_available = available_features(df, CATEGORICAL_FEATURES)

st.title("Laboratorio interactivo de clasificación ML")
st.caption(source_label)
for warning in dataset_warnings:
    st.warning(warning)

intro_left, intro_right = st.columns([1.2, 1])
with intro_left:
    st.markdown(
        """
        <div class="concept-box">
        <b>Objetivo:</b> jugar con variables, normalización, split, modelo y balanceo para ver cómo cambian los resultados.
        <br><br>
        Queremos predecir si una venta será devuelta usando información como canal, país, descuento,
        días de entrega, categoría, margen, stock y rating del cliente.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    c3.metric("Variables numéricas", len(numeric_available))
    c4.metric("Variables categóricas", len(categorical_available))
with intro_right:
    st.plotly_chart(target_distribution(df), width="stretch", key="target_distribution_top")

section("1. Mira los datos antes de elegir variables")
explore_left, explore_right = st.columns([1.15, 1])
with explore_left:
    st.write("**Matriz de correlación de variables numéricas**")
    st.plotly_chart(correlation_heatmap(df), width="stretch", key="correlation_before_selection")
with explore_right:
    st.write("**Varianza y escala de cada variable numérica**")
    variance_df = variance_table(df, numeric_available)
    st.dataframe(variance_df, width="stretch", height=430)
    st.write(
        "Una variable con varianza muy baja cambia poco. Una variable con escala enorme puede dominar modelos como KNN si no normalizamos."
    )

with st.expander("Ver primeras filas, nulos y tipos de datos", expanded=False):
    d1, d2 = st.columns([1.2, 1])
    d1.dataframe(df.head(20), width="stretch")
    d2.plotly_chart(missing_values(df), width="stretch", key="missing_values")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={"index": "columna", 0: "tipo"}), width="stretch")

pairs = high_correlation_pairs(df, threshold=0.85)
st.write("**Pares con correlación alta (> 0.85)**")
st.dataframe(
    pairs if not pairs.empty else pd.DataFrame({"mensaje": ["No se encontraron pares sobre 0.85."]}),
    width="stretch",
)
questions(
    [
        "¿Qué variables parecen repetidas?",
        "¿Qué variables tienen una escala mucho más grande?",
        "¿Qué variables crees que ayudarán a predecir devoluciones?",
    ]
)

section("2. Configura tu experimento")
st.write("Elige variables y parámetros. El modelo no corre hasta que presiones **Correr experimento**.")

config_left, config_mid, config_right = st.columns([1.1, 1, 1])
with config_left:
    st.write("**Variables predictoras**")
    variable_preset = st.radio(
        "Selección rápida",
        ["Todas", "Solo numéricas", "Solo categóricas", "Personalizado"],
        horizontal=True,
    )
    if variable_preset == "Todas":
        default_num = numeric_available
        default_cat = categorical_available
    elif variable_preset == "Solo numéricas":
        default_num = numeric_available
        default_cat = []
    elif variable_preset == "Solo categóricas":
        default_num = []
        default_cat = categorical_available
    else:
        default_num = ["rating_cliente", "descuento", "venta_neta", "margen_gtq", "dias_entrega", "entrega_tardia", "stock_disponible"]
        default_num = [col for col in default_num if col in numeric_available]
        default_cat = ["pais", "canal", "categoria", "metodo_pago", "promocion", "prioridad_envio"]
        default_cat = [col for col in default_cat if col in categorical_available]

    selected_numeric = st.multiselect(
        "Variables numéricas",
        numeric_available,
        default=default_num,
        help="Puedes elegir una, varias o ninguna.",
    )
    selected_categorical = st.multiselect(
        "Variables categóricas",
        categorical_available,
        default=default_cat,
        help="Se transforman con OneHotEncoder.",
    )
    include_redundant = st.checkbox("Agregar variables redundantes/correlacionadas", value=False)

with config_mid:
    st.write("**Split y preprocesamiento**")
    train_pct = st.slider("% entrenamiento", 50, 90, 80, 5)
    test_size = (100 - train_pct) / 100
    st.plotly_chart(split_bar(train_pct, 100 - train_pct), width="stretch", key="split_config")
    scaling_choice = st.radio(
        "Técnica de normalizado",
        ["Sin normalización", "StandardScaler"],
        index=1,
        help="StandardScaler deja media cercana a 0 y desviación estándar cercana a 1.",
    )
    use_scaling = scaling_choice == "StandardScaler"
    balance_label = st.radio(
        "Balanceo de datos",
        ["Sin balanceo", "Class weight balanced", "Sobremuestreo clase minoritaria"],
        index=1,
    )
    balance_strategy = balance_label_to_strategy(balance_label)
    random_state = st.number_input("Random state", value=42, step=1)

with config_right:
    st.write("**Modelo y parámetros**")
    model_name = st.selectbox("Modelo", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"], index=3)
    model_params = {}
    if model_name == "Random Forest":
        model_params["n_estimators"] = st.slider("n_estimators", 50, 400, 150, 10)
        rf_depth = st.slider("max_depth", 2, 30, 10, 1)
        model_params["max_depth"] = None if st.checkbox("Sin límite de profundidad") else rf_depth
        model_params["min_samples_leaf"] = st.slider("min_samples_leaf RF", 1, 20, 1, 1)
    elif model_name == "Decision Tree":
        model_params["max_depth"] = st.slider("max_depth", 1, 30, 6, 1)
        model_params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 20, 3, 1)
    elif model_name == "KNN":
        model_params["n_neighbors"] = st.slider("n_neighbors", 1, 25, 5, 1)
        if balance_strategy == "class_weight":
            st.warning("KNN no usa class_weight. Prueba sobremuestreo si quieres balancearlo.")
    else:
        model_params["C"] = st.slider("C", 0.05, 5.0, 1.0, 0.05)

df_model = add_redundant_features(df, int(random_state)) if include_redundant else df.copy()
redundant_cols = [col for col in df_model.columns if col.endswith("_dup")]
selected_features = selected_numeric + selected_categorical + redundant_cols

summary_cols = st.columns(5)
summary_cols[0].metric("Features elegidas", len(selected_features))
summary_cols[1].metric("Numéricas", len(selected_numeric) + len(redundant_cols))
summary_cols[2].metric("Categóricas", len(selected_categorical))
summary_cols[3].metric("Train", f"{train_pct}%")
summary_cols[4].metric("Test", f"{100 - train_pct}%")

with st.expander("Ver variables seleccionadas y su correlación", expanded=True):
    st.write(selected_features if selected_features else "No has elegido variables.")
    selected_numeric_for_corr = [col for col in selected_numeric + redundant_cols if col in df_model.columns]
    if selected_numeric_for_corr:
        st.plotly_chart(correlation_heatmap(df_model[selected_numeric_for_corr + [TARGET]]), width="stretch", key="selected_corr")
    if selected_numeric:
        selected_variance = variance_table(df_model, [col for col in selected_numeric if col in df_model.columns])
        st.dataframe(selected_variance, width="stretch")

run_clicked = st.button("Correr experimento", type="primary", disabled=not selected_features)
if run_clicked:
    with st.spinner("Entrenando con las variables y parámetros elegidos..."):
        try:
            result = cached_train(
                df_model,
                selected_features,
                model_name,
                use_scaling,
                float(test_size),
                int(random_state),
                model_params,
                balance_strategy,
            )
            st.session_state["last_result"] = result
            st.session_state["last_config"] = {
                "features": selected_features,
                "model_name": model_name,
                "use_scaling": use_scaling,
                "test_size": test_size,
                "random_state": int(random_state),
                "model_params": model_params,
                "balance_strategy": balance_strategy,
                "balance_label": balance_label,
                "include_redundant": include_redundant,
                "df_model": df_model,
            }
            st.session_state["last_error"] = None
        except Exception as exc:
            st.session_state["last_result"] = None
            st.session_state["last_error"] = str(exc)

if st.session_state.get("last_error"):
    st.error(f"No se pudo entrenar: {st.session_state['last_error']}")

result = st.session_state.get("last_result")
config = st.session_state.get("last_config")

section("3. Resultados del último experimento")
if result is None:
    st.info("Ajusta las opciones y presiona **Correr experimento** para ver resultados.")
else:
    cfg_text = (
        f"Modelo: **{config['model_name']}** · "
        f"Normalización: **{'StandardScaler' if config['use_scaling'] else 'No'}** · "
        f"Balanceo: **{config['balance_label']}** · "
        f"Features: **{len(config['features'])}** · "
        f"Test: **{config['test_size']:.0%}**"
    )
    st.markdown(cfg_text)
    metric_row(result.metrics)

    b1, b2 = st.columns(2)
    before_balance = result.metrics["train_distribution_before"]
    after_balance = result.metrics["train_distribution_after"]
    balance_df = pd.DataFrame(
        [
            {"momento": "Antes de balancear", "clase": str(k), "filas": v}
            for k, v in before_balance.items()
        ]
        + [
            {"momento": "Después de balancear", "clase": str(k), "filas": v}
            for k, v in after_balance.items()
        ]
    )
    with b1:
        st.write("**Distribución del entrenamiento antes/después del balanceo**")
        st.plotly_chart(px.bar(balance_df, x="clase", y="filas", color="momento", barmode="group"), width="stretch", key="balance_plot")
    with b2:
        st.write("**Matriz de confusión**")
        st.plotly_chart(confusion_matrix_plot(result.confusion), width="stretch", key="confusion_results")

    st.write(
        "**Cómo leerlo:** si sube el recall, el modelo detecta más devoluciones reales. "
        "Pero si baja mucho la precision, también puede estar marcando demasiadas ventas como devolución."
    )

    r1, r2 = st.columns([1, 1])
    with r1:
        if result.roc is not None:
            st.write("**Curva ROC**")
            st.plotly_chart(roc_curve_plot(result.roc), width="stretch", key="roc_results")
    with r2:
        st.write("**Classification report**")
        st.code(result.report)

    try:
        importance = compute_feature_importance(result, int(config["random_state"]))
        st.write("**Top 10 variables más importantes**")
        i1, i2 = st.columns([1.2, 1])
        i1.plotly_chart(feature_importance_plot(importance), width="stretch", key="importance_results")
        i2.dataframe(importance, width="stretch")
    except Exception as exc:
        st.warning(f"No se pudo calcular importancia de variables: {exc}")

questions(
    [
        "¿Cambió más accuracy, recall o precision?",
        "¿El balanceo ayudó a detectar más devoluciones?",
        "¿El modelo generaliza o solo parece bueno en una métrica?",
    ]
)

section("4. Usar el modelo para predecir una venta nueva")
if result is None or config is None:
    st.info("Primero corre un experimento. La predicción usará exactamente ese modelo entrenado.")
else:
    st.markdown(
        """
        En la vida real, después de entrenar y validar un modelo, se usa así:

        1. Llega una venta nueva con datos conocidos al momento de vender: país, canal, descuento, días estimados, stock, etc.
        2. Esos valores pasan por el mismo preprocesamiento usado en entrenamiento: imputación, OneHotEncoder y normalización si aplica.
        3. El modelo calcula una probabilidad o una clase.
        4. El negocio usa esa predicción para decidir una acción: revisar la orden, priorizar atención, evitar una devolución o estimar riesgo.

        Importante: no se deben usar columnas que solo se conocen después, como motivo o fecha de devolución.
        """
    )

    model_used = result.pipeline.named_steps["model"]
    st.write("**Cómo decide este modelo**")
    if config["model_name"] == "Logistic Regression":
        equation, coefficients = logistic_equation(result)
        st.write(
            "Logistic Regression sí tiene una ecuación. La ecuación calcula un puntaje llamado `logit(p)` "
            "y luego lo convierte en probabilidad con `p = 1 / (1 + exp(-logit))`."
        )
        st.code(equation)
        with st.expander("Ver coeficientes completos", expanded=False):
            st.dataframe(coefficients.drop(columns=["impacto_abs"]), width="stretch")
        st.caption(
            "Coeficientes positivos aumentan la probabilidad de devolución; coeficientes negativos la reducen. "
            "Si usaste StandardScaler, la ecuación usa variables normalizadas, no los números originales directamente."
        )
    elif config["model_name"] == "Decision Tree":
        st.write(
            "Decision Tree no usa una sola ecuación lineal. Usa reglas tipo `si descuento > x y entrega_tardia <= y, entonces...`."
        )
        with st.expander("Ver reglas principales del árbol", expanded=False):
            st.code(decision_tree_rules(result, max_depth=4))
    elif config["model_name"] == "Random Forest":
        st.write(
            "Random Forest no tiene una sola ecuación final: entrena muchos árboles y combina sus votos. "
            "Por eso suele ser potente, pero menos fácil de explicar que una regresión logística o un árbol simple."
        )
    else:
        st.write(
            "KNN no aprende una ecuación. Para una venta nueva, busca ventas parecidas en el entrenamiento y vota según sus vecinos. "
            "Por eso la escala de las variables puede cambiar mucho el resultado."
        )

    st.write("**Simulador de predicción**")
    with st.form("prediction_form"):
        st.write("Edita la fila como si fuera una venta nueva. Cada columna es una variable del modelo.")
        new_row = st.data_editor(
            default_prediction_values(
                config["df_model"],
                config["features"],
                NUMERIC_FEATURES + [col for col in config["features"] if col.endswith("_dup")],
            ),
            column_config=prediction_column_config(
                config["df_model"],
                config["features"],
                NUMERIC_FEATURES + [col for col in config["features"] if col.endswith("_dup")],
            ),
            hide_index=True,
            num_rows="fixed",
            width="stretch",
            key="prediction_values_editor",
        )
        predict_clicked = st.form_submit_button("Predecir con estos valores", type="primary")

    if predict_clicked:
        try:
            pred_class = int(result.pipeline.predict(new_row)[0])
            pred_label = "Devuelto" if pred_class == 1 else "No devuelto"
            pred_proba = None
            if hasattr(result.pipeline, "predict_proba"):
                pred_proba = float(result.pipeline.predict_proba(new_row)[0, 1])

            p1, p2, p3 = st.columns(3)
            p1.metric("Predicción", pred_label)
            p2.metric("Clase", pred_class)
            p3.metric("Probabilidad de devolución", "N/A" if pred_proba is None else f"{pred_proba:.1%}")

            st.dataframe(new_row, width="stretch")
            st.write(
                "Esta es la misma idea que se implementa en producción: una aplicación, API o proceso automático recibe datos nuevos, "
                "aplica el mismo pipeline y devuelve una predicción para apoyar una decisión."
            )
        except Exception as exc:
            st.error(f"No se pudo predecir con estos valores: {exc}")

section("5. Comparaciones rápidas")
if config is None:
    st.info("Corre primero un experimento para activar comparaciones.")
else:
    comp_left, comp_right = st.columns(2)
    with comp_left:
        st.write("**Con vs. sin variables redundantes**")
        try:
            comparison = compare_redundant_features(
                df,
                add_redundant_features(df, int(config["random_state"])),
                [col for col in config["features"] if not col.endswith("_dup")],
                config["model_name"],
                config["use_scaling"],
                float(config["test_size"]),
                int(config["random_state"]),
                balance_strategy=config["balance_strategy"],
            )
            st.plotly_chart(metrics_comparison(comparison), width="stretch", key="redundant_comparison")
            st.dataframe(comparison, width="stretch")
        except Exception as exc:
            st.warning(f"No se pudo comparar redundancia: {exc}")

    with comp_right:
        st.write("**Con vs. sin normalización**")
        try:
            res_no = cached_train(
                config["df_model"],
                config["features"],
                config["model_name"],
                False,
                float(config["test_size"]),
                int(config["random_state"]),
                config["model_params"],
                config["balance_strategy"],
            )
            res_yes = cached_train(
                config["df_model"],
                config["features"],
                config["model_name"],
                True,
                float(config["test_size"]),
                int(config["random_state"]),
                config["model_params"],
                config["balance_strategy"],
            )
            norm_df = pd.DataFrame(
                [
                    {"experimento": "Sin normalización", **{k: res_no.metrics[k] for k in ["accuracy", "precision", "recall", "f1"]}},
                    {"experimento": "Con StandardScaler", **{k: res_yes.metrics[k] for k in ["accuracy", "precision", "recall", "f1"]}},
                ]
            )
            st.plotly_chart(metrics_comparison(norm_df), width="stretch", key="normalization_comparison")
            st.dataframe(norm_df, width="stretch")
        except Exception as exc:
            st.warning(f"No se pudo comparar normalización: {exc}")

section("6. Normalización visual")
st.write(
    "StandardScaler transforma variables numéricas para que estén en escalas comparables. "
    "Esto suele importar más en KNN y Logistic Regression que en árboles."
)
before, after = scaling_stats(df, ["precio_unitario", "venta_neta", "rating_cliente", "dias_entrega"])
if not before.empty:
    n1, n2 = st.columns(2)
    n1.dataframe(before, width="stretch")
    n2.dataframe(after, width="stretch")
    st.plotly_chart(before_after_scaling(before, after), width="stretch", key="scaling_visual")

section("7. Overfitting con Decision Tree")
if config is None:
    st.info("Corre primero un experimento.")
else:
    max_depth_limit = st.slider("Profundidad máxima para probar overfitting", 1, 20, 20, 1)
    leaf_value = st.slider("min_samples_leaf para overfitting", 1, 20, 3, 1)
    try:
        curve = overfitting_curve(
            config["df_model"],
            config["features"],
            config["use_scaling"],
            float(config["test_size"]),
            int(config["random_state"]),
            int(leaf_value),
            int(max_depth_limit),
            balance_strategy=config["balance_strategy"],
        )
        o1, o2 = st.columns([1.2, 1])
        o1.plotly_chart(overfitting_line(curve), width="stretch", key="overfitting_curve")
        best_row = curve.sort_values("accuracy_test", ascending=False).iloc[0]
        o2.metric("Mejor profundidad test", int(best_row["max_depth"]))
        o2.metric("Mejor accuracy test", f"{best_row['accuracy_test']:.3f}")
        o2.dataframe(curve, width="stretch", height=320)
    except Exception as exc:
        st.warning(f"No se pudo calcular overfitting: {exc}")

section("8. Posible data leakage")
st.write(
    "Data leakage ocurre cuando usamos una variable que no deberíamos conocer al momento de predecir. "
    "Por ejemplo, `fecha_devolucion` o `motivo_devolucion` revelan información posterior a la venta."
)
suspicious = find_leakage_columns(df)
if suspicious:
    st.error("Cuidado: estas columnas podrían revelar la respuesta y hacer que el modelo parezca mejor de lo real.")
    st.write(suspicious)
else:
    st.success("No se encontraron columnas sospechosas por nombre.")

section("9. Retos para experimentar")
challenge_cols = st.columns(2)
challenges = [
    ("Solo numéricas", "Corre un modelo solo con variables numéricas. ¿Qué desempeño obtuviste?"),
    ("Numéricas + categóricas", "Agrega categóricas. ¿Mejoró el resultado?"),
    ("Correlacionadas", "Activa variables redundantes. ¿Mejoró o solo aumentó complejidad?"),
    ("Balanceo", "Compara sin balanceo vs sobremuestreo. ¿Qué pasó con recall?"),
    ("Normalización", "Usa KNN con y sin StandardScaler. ¿Qué cambió?"),
    ("Split", "Cambia train de 80% a 60%. ¿Cambian mucho las métricas?"),
    ("Overfitting", "Sube max_depth en Decision Tree. ¿Train sube más que test?"),
    ("Importancia", "Identifica las 5 variables más importantes. ¿Tiene sentido de negocio?"),
]
for idx, (title, text) in enumerate(challenges):
    with challenge_cols[idx % 2]:
        done = st.checkbox(title, key=f"challenge_{idx}")
        st.caption(text if not done else f"Completado. {text}")
