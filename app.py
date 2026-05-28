from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text
from streamlit_sortables import sort_items

from src.modeling import (
    compare_redundant_features,
    compute_feature_importance,
    overfitting_curve,
    train_model,
)
from src.movie_regression import (
    MANUAL_MOVIE_TARGET,
    MULTIPLE_FEATURES,
    SIMPLE_FEATURE,
    compute_errors,
    compute_gradients_multiple,
    compute_gradients_simple,
    compute_mse,
    generate_large_movie_dataset,
    normalize_features,
    predict_multiple,
    predict_simple,
    simple_metrics,
    sklearn_comparison,
    small_manual_movie_dataset,
    train_multiple_steps,
    train_simple_steps,
    update_multiple_params,
    update_simple_params,
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
    .chip-container {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;}
    .chip {
        display: inline-block;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 7px 11px;
        color: #1f2937;
        font-size: .92rem;
        font-weight: 600;
    }
    .badge-ok {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: .88rem;
    }
    .badge-warn {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: .88rem;
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


def flow_steps() -> list[dict]:
    return [
        {
            "title": "DWH",
            "icon": "🏢",
            "short": "Fuente histórica de datos",
            "why": "Centraliza ventas, clientes, productos y devoluciones para analizarlos.",
            "input": ["Ventas históricas", "Clientes", "Productos", "Devoluciones"],
            "process": ["Almacenar información", "Consolidar datos históricos"],
            "output": ["Datos disponibles para análisis"],
            "example": ["ventas_2025", "clientes", "productos", "devoluciones"],
        },
        {
            "title": "Dataset",
            "icon": "🧾",
            "short": "Tabla o CSV para entrenar",
            "why": "Convierte muchas tablas en una tabla de trabajo con filas y columnas.",
            "input": ["Tablas o consulta SQL"],
            "process": ["Extraer columnas relevantes", "Convertir a CSV o tabla de trabajo"],
            "output": ["Dataset con filas y columnas para modelar"],
            "example": ["1 fila = 1 venta", "columnas = pais, descuento, devuelto"],
        },
        {
            "title": "Limpieza",
            "icon": "🧹",
            "short": "Preparar variables",
            "why": "El modelo necesita datos consistentes y sin errores obvios.",
            "input": ["Dataset crudo"],
            "process": ["convertir tipos", "imputar nulos", "quitar errores", "seleccionar features", "separar target"],
            "output": ["X (features)", "y (target)", "datos listos para modelar"],
            "example": ['pais = "Guate " -> "Guatemala"', 'descuento = "0.1" -> 0.10', "target = devuelto"],
        },
        {
            "title": "Train/Test",
            "icon": "✂️",
            "short": "Separar para aprender y evaluar",
            "why": "Simula datos no vistos para medir si el modelo generaliza.",
            "input": ["X", "y"],
            "process": ["dividir datos"],
            "output": ["X_train", "X_test", "y_train", "y_test"],
            "example": ["1000 filas", "800 train", "200 test"],
        },
        {
            "title": "Modelo",
            "icon": "🧠",
            "short": "Aprender patrones",
            "why": "Encuentra relaciones entre features y devoluciones.",
            "input": ["X_train", "y_train"],
            "process": ["entrenar algoritmo"],
            "output": ["modelo entrenado"],
            "example": ["Random Forest aprende reglas", "Logistic Regression aprende coeficientes"],
        },
        {
            "title": "Resultados",
            "icon": "📈",
            "short": "Medir desempeño",
            "why": "Ayuda a decidir si el modelo sirve antes de usarlo en la vida real.",
            "input": ["modelo", "X_test", "y_test"],
            "process": ["generar predicciones", "calcular métricas"],
            "output": ["accuracy", "precision", "recall", "F1", "matriz de confusión"],
            "example": ["recall = 0.72", "F1 = 0.68", "matriz de confusión"],
        },
        {
            "title": "Predicción",
            "icon": "🔮",
            "short": "Predecir venta nueva",
            "why": "Usa lo aprendido para estimar el riesgo de una venta nueva.",
            "input": ["una venta nueva"],
            "process": ["pasarla por el mismo pipeline", "aplicar modelo"],
            "output": ["predicción 0/1", "probabilidad"],
            "example": ["canal = Online", "descuento = 0.25", "dias_entrega = 8", "probabilidad devolución = 0.72"],
        },
        {
            "title": "Decisión",
            "icon": "✅",
            "short": "Actuar con el resultado",
            "why": "Convierte la predicción en una acción de negocio.",
            "input": ["predicción del modelo"],
            "process": ["interpretar resultado"],
            "output": ["priorizar revisión", "alertar riesgo", "tomar acción de negocio"],
            "example": ["si riesgo > 70%", "revisar orden", "contactar cliente"],
        },
    ]


def render_flow_diagram(steps: list[dict]) -> None:
    for idx, step in enumerate(steps):
        with st.container(border=True):
            c1, c2 = st.columns([0.12, 0.88])
            c1.markdown(f"### {step['icon']}")
            c2.markdown(f"**{idx + 1}. {step['title']}**")
            c2.caption(step["short"])
        if idx < len(steps) - 1:
            st.markdown("<div style='text-align:center;color:#64748b;font-size:1.4rem;'>↓</div>", unsafe_allow_html=True)


def render_chips(items: list[str], color: str = "#eef6ff") -> None:
    chips = "".join(
        f"<span class='chip' style='background:{color};'>{str(item)}</span>"
        for item in items
    )
    st.markdown(f"<div class='chip-container'>{chips}</div>", unsafe_allow_html=True)


def render_io_block(title: str, icon: str, items: list[str], color: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        render_chips(items, color=color)


def render_stage_io_card(stage: dict) -> None:
    st.markdown(f"### {stage['icon']} {stage['title']}")
    st.write(stage["why"])
    c1, c2, c3 = st.columns(3)
    with c1:
        render_io_block("Entrada", "📥", stage["input"], "#eff6ff")
    with c2:
        render_io_block("Proceso", "⚙️", stage["process"], "#f8fafc")
    with c3:
        render_io_block("Salida", "📤", stage["output"], "#f0fdf4")

    st.write("**Ejemplo rápido**")
    render_chips(stage["example"], color="#fff7ed")


def render_flow_explanation(steps: list[dict]) -> None:
    for step in steps:
        with st.expander(f"{step['icon']} {step['title']}"):
            render_stage_io_card(step)


def initial_flow_containers(steps: list[dict]) -> list[dict]:
    shuffled = ["Modelo", "DWH", "Predicción", "Limpieza", "Resultados", "Dataset", "Decisión", "Train/Test"]
    available = [item for item in shuffled if item in [step["title"] for step in steps]]
    return [
        {"header": "Etapas disponibles", "items": available},
        {"header": "Tu flujo", "items": []},
    ]


def validate_flow_order(user_order: list[str], correct_order: list[str]) -> tuple[int, pd.DataFrame]:
    rows = []
    for idx, expected in enumerate(correct_order):
        selected = user_order[idx] if idx < len(user_order) else "(vacío)"
        ok = selected == expected
        rows.append(
            {
                "posición": idx + 1,
                "tu_etapa": selected,
                "etapa_correcta": expected,
                "estado": "Correcto" if ok else "Revisar",
            }
        )
    return sum(row["estado"] == "Correcto" for row in rows), pd.DataFrame(rows)


def render_flow_sortable_challenge(steps: list[dict]) -> None:
    correct_order = [step["title"] for step in steps]
    if "flow_reset_counter" not in st.session_state:
        st.session_state.flow_reset_counter = 0
    if "flow_validation" not in st.session_state:
        st.session_state.flow_validation = None

    st.write("Pista: primero necesitamos datos históricos, luego limpiarlos, después entrenar, evaluar y finalmente usar el modelo.")
    custom_style = """
    .sortable-component {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 8px;
        background: #ffffff;
    }
    .sortable-container {
        background-color: #f8fafc;
        border-radius: 10px;
        min-height: 260px;
    }
    .sortable-container-header {
        background-color: #e0f2fe;
        color: #0f172a;
        font-weight: 700;
        border-radius: 8px 8px 0 0;
        padding: .55rem .75rem;
    }
    .sortable-item, .sortable-item:hover {
        background-color: #ffffff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        color: #1f2937;
        font-weight: 650;
        margin: 6px;
        padding: 10px;
    }
    """
    containers = sort_items(
        initial_flow_containers(steps),
        multi_containers=True,
        direction="horizontal",
        custom_style=custom_style,
        key=f"flow_sortable_{st.session_state.flow_reset_counter}",
    )
    st.session_state.flow_current_containers = containers
    user_flow = containers[1]["items"] if len(containers) > 1 else []

    b1, b2 = st.columns([1, 1])
    if b1.button("Validar flujo", type="primary"):
        score, feedback = validate_flow_order(user_flow, correct_order)
        st.session_state.flow_validation = {"score": score, "feedback": feedback}
        st.session_state.flow_progress = score
    if b2.button("Reintentar"):
        st.session_state.flow_reset_counter += 1
        st.session_state.flow_validation = None
        st.rerun()

    validation = st.session_state.flow_validation
    if validation:
        score = validation["score"]
        st.metric("Puntaje", f"{score}/8")
        st.progress(score / 8)
        styled = validation["feedback"].style.apply(
            lambda row: ["background-color: #dcfce7" if row["estado"] == "Correcto" else "background-color: #fef3c7"] * len(row),
            axis=1,
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        if score >= 7:
            st.success("Muy bien, ya entendés casi todo el flujo.")
        elif score >= 4:
            st.info("Vas bien. Revisá qué pasa antes del modelo y qué pasa después de resultados.")
        else:
            st.warning("Revisá el orden: datos históricos → limpieza → entrenamiento/evaluación → uso en producción.")


def render_stage_io(steps: list[dict]) -> None:
    selected = st.selectbox("Elige una etapa", [step["title"] for step in steps], key="flow_stage_selector")
    step = next(item for item in steps if item["title"] == selected)
    render_stage_io_card(step)


def render_training_vs_production() -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="concept-box">
            <h4>Entrenamiento</h4>
            <ul>
            <li>Tenemos datos históricos.</li>
            <li>Sí conocemos el target <code>devuelto</code>.</li>
            <li>Dividimos train/test.</li>
            <li>Medimos métricas.</li>
            <li>Ajustamos parámetros.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="concept-box">
            <h4>Producción</h4>
            <ul>
            <li>Llega una venta nueva.</li>
            <li>No conocemos el target real todavía.</li>
            <li>Usamos el pipeline ya entrenado.</li>
            <li>Hacemos una predicción.</li>
            <li>Usamos la predicción para decidir.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.warning("En entrenamiento el modelo aprende. En producción el modelo ya no está aprendiendo, solo está usando lo que aprendió.")


def render_complete_flow_challenge(steps: list[dict]) -> None:
    st.write("Completa los espacios vacíos del flujo. Algunas etapas ya están fijas para guiarte.")
    fixed_positions = {0: "DWH", 2: "Limpieza", 4: "Modelo", 7: "Decisión"}
    options = ["Selecciona..."] + [step["title"] for step in steps]
    score = 0
    answers = {}

    for idx, step in enumerate(steps):
        with st.container(border=True):
            if idx in fixed_positions:
                st.markdown(f"**{idx + 1}. {step['icon']} {step['title']}**")
                st.caption(step["short"])
                st.markdown("<span class='badge-ok'>Etapa fija</span>", unsafe_allow_html=True)
                score += 1
            else:
                selected = st.selectbox(
                    f"Paso {idx + 1}",
                    options,
                    key=f"complete_flow_step_{idx}",
                )
                answers[idx] = selected
                if selected != "Selecciona...":
                    chosen_step = next((item for item in steps if item["title"] == selected), None)
                    if chosen_step:
                        st.caption(chosen_step["short"])
                if selected == step["title"]:
                    st.markdown("<span class='badge-ok'>Correcto</span>", unsafe_allow_html=True)
                    score += 1
                elif selected != "Selecciona...":
                    st.markdown("<span class='badge-warn'>Revisar posición</span>", unsafe_allow_html=True)

        if idx < len(steps) - 1:
            st.markdown("<div style='text-align:center;color:#64748b;font-size:1.4rem;'>↓</div>", unsafe_allow_html=True)

    if st.button("Validar flujo completado", type="primary"):
        st.session_state.complete_flow_score = score
        st.session_state.complete_flow_answers = answers

    if "complete_flow_score" in st.session_state:
        final_score = st.session_state.complete_flow_score
        st.metric("Puntaje del flujo", f"{final_score}/8")
        st.progress(final_score / 8)
        if final_score == 8:
            st.success("Flujo completo. Ya entendés el viaje desde datos históricos hasta decisión.")
        elif final_score >= 5:
            st.info("Vas bien. Revisa qué pasa antes del modelo y qué pasa después de resultados.")
        else:
            st.warning("Repasa el orden completo: DWH → Dataset → Limpieza → Train/Test → Modelo → Resultados → Predicción → Decisión.")


def render_model_data_flow_section() -> None:
    steps = flow_steps()
    section(
        "Flujo de datos del modelo",
        "Mini laboratorio visual para entender cómo los datos históricos terminan en una predicción real.",
    )
    flow_tab, sort_tab, io_tab, complete_tab = st.tabs(
        ["Vista completa", "Ordena el flujo", "Entrada / Proceso / Salida", "Completa el flujo"]
    )
    with flow_tab:
        render_flow_diagram(steps)
        if st.button("Ver explicación del flujo"):
            st.session_state.show_flow_explanation = not st.session_state.get("show_flow_explanation", False)
        if st.session_state.get("show_flow_explanation", False):
            render_flow_explanation(steps)
        st.subheader("Entrenamiento vs Producción")
        render_training_vs_production()
    with sort_tab:
        render_flow_sortable_challenge(steps)
    with io_tab:
        render_stage_io(steps)
    with complete_tab:
        render_complete_flow_challenge(steps)


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


@st.cache_data(show_spinner=False)
def cached_movie_data(n_rows: int, random_state: int) -> tuple[pd.DataFrame, str]:
    return load_or_generate_movie_dataset(n_rows=n_rows, random_state=random_state)


@st.cache_resource(show_spinner=False)
def cached_movie_train(
    df: pd.DataFrame,
    selected_features: tuple[str, ...],
    use_scaling: bool,
    test_size: float,
    random_state: int,
):
    return train_movie_regression(df, list(selected_features), use_scaling, test_size, random_state)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def simple_regression_fit(df: pd.DataFrame, feature: str) -> tuple[LinearRegression, pd.DataFrame, dict]:
    data = df[[feature, MOVIE_TARGET]].dropna().copy()
    X = data[[feature]]
    y = data[MOVIE_TARGET]
    model = LinearRegression()
    model.fit(X, y)
    data["prediccion"] = np.clip(model.predict(X), 1.0, 10.0)
    metrics = {
        "MAE": mean_absolute_error(y, data["prediccion"]),
        "RMSE": rmse(y, data["prediccion"]),
        "R2": r2_score(y, data["prediccion"]),
    }
    return model, data, metrics


def simple_regression_plot(data: pd.DataFrame, feature: str, model: LinearRegression, show_errors: bool = False) -> go.Figure:
    fig = px.scatter(
        data,
        x=feature,
        y=MOVIE_TARGET,
        opacity=0.65,
        labels={MOVIE_TARGET: "rating_numerico", feature: feature},
        color_discrete_sequence=["#457b9d"],
    )
    x_line = np.linspace(data[feature].min(), data[feature].max(), 100)
    y_line = np.clip(model.predict(pd.DataFrame({feature: x_line})), 1.0, 10.0)
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Linea de regresion", line=dict(color="#e76f51", width=3)))
    if show_errors:
        sample = data.sample(min(18, len(data)), random_state=7).sort_values(feature)
        for _, row in sample.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row[feature], row[feature]],
                    y=[row[MOVIE_TARGET], row["prediccion"]],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="#111827", width=1, dash="dot"),
                    hoverinfo="skip",
                )
            )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def rating_real_vs_pred_plot(y_true, y_pred) -> go.Figure:
    plot_df = pd.DataFrame({"rating real": y_true, "rating predicho": y_pred})
    fig = px.scatter(plot_df, x="rating real", y="rating predicho", opacity=0.7, color_discrete_sequence=["#2a9d8f"])
    fig.add_trace(go.Scatter(x=[1, 10], y=[1, 10], mode="lines", name="Ideal y=x", line=dict(color="#e76f51", dash="dash")))
    fig.update_layout(height=420, xaxis=dict(range=[1, 10]), yaxis=dict(range=[1, 10]))
    return fig


def residual_plots(y_true, y_pred) -> tuple[go.Figure, go.Figure]:
    errors = pd.Series(y_true.to_numpy() - np.asarray(y_pred), name="error")
    hist = px.histogram(errors.to_frame(), x="error", nbins=30, color_discrete_sequence=["#457b9d"])
    hist.update_layout(height=360)
    resid = px.scatter(
        pd.DataFrame({"rating predicho": y_pred, "error": errors}),
        x="rating predicho",
        y="error",
        opacity=0.7,
        color_discrete_sequence=["#e76f51"],
    )
    resid.add_hline(y=0, line_dash="dash", line_color="#111827")
    resid.update_layout(height=360)
    return hist, resid


def train_test_metrics_row(metrics: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("MAE train", f"{metrics['mae_train']:.3f}")
    cols[1].metric("MAE test", f"{metrics['mae_test']:.3f}")
    cols[2].metric("RMSE train", f"{metrics['rmse_train']:.3f}")
    cols[3].metric("RMSE test", f"{metrics['rmse_test']:.3f}")
    cols[4].metric("R2 train", f"{metrics['r2_train']:.3f}")
    cols[5].metric("R2 test", f"{metrics['r2_test']:.3f}")


def normalization_comparison_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(values)
    scaled = StandardScaler().fit_transform(imputed)
    return pd.DataFrame(
        {
            "Variable": cols,
            "media antes": imputed.mean(axis=0),
            "std antes": imputed.std(axis=0),
            "media despues": scaled.mean(axis=0),
            "std despues": scaled.std(axis=0),
        }
    )


def movie_model_comparison(df: pd.DataFrame, test_size: float, random_state: int, use_scaling: bool) -> pd.DataFrame:
    configs = [
        ("Modelo A", ["reseñas_positivas_pct"]),
        ("Modelo B", ["reseñas_positivas_pct", "popularidad"]),
        (
            "Modelo C",
            [
                "reseñas_positivas_pct",
                "popularidad",
                "votos_usuarios",
                "director_experiencia_anios",
                "presupuesto_millones",
                "marketing_millones",
            ],
        ),
        ("Modelo D", MOVIE_FEATURES),
    ]
    rows = []
    for name, features in configs:
        result = train_movie_regression(df, features, use_scaling, test_size, random_state)
        rows.append(
            {
                "Modelo": name,
                "Features usadas": ", ".join(features),
                "MAE test": result.metrics["mae_test"],
                "RMSE test": result.metrics["rmse_test"],
                "R2 test": result.metrics["r2_test"],
            }
        )
    return pd.DataFrame(rows)


def overfitting_regression_comparison(df: pd.DataFrame, test_size: float, random_state: int, use_scaling: bool, add_noise: bool) -> pd.DataFrame:
    data_dup = add_correlated_movie_features(df, random_state)
    data_noise = add_noise_movie_features(data_dup if add_noise else df, random_state)
    configs = [
        ("Simple", df, ["reseñas_positivas_pct", "popularidad"]),
        ("Muchas variables", df, MOVIE_FEATURES),
        ("Duplicadas", data_dup, MOVIE_FEATURES + ["popularidad_dup", "reseñas_positivas_dup", "marketing_dup"]),
    ]
    if add_noise:
        configs.append(("Con ruido", data_noise, MOVIE_FEATURES + [f"ruido_{idx}" for idx in range(1, 6)]))
    rows = []
    for name, data, features in configs:
        result = train_movie_regression(data, features, use_scaling, test_size, random_state)
        rows.append(
            {
                "Modelo": name,
                "MAE train": result.metrics["mae_train"],
                "MAE test": result.metrics["mae_test"],
                "R2 train": result.metrics["r2_train"],
                "R2 test": result.metrics["r2_test"],
            }
        )
    return pd.DataFrame(rows)


def render_movie_quiz() -> None:
    quiz = [
        ("¿Qué predice una regresión lineal?", ["Un número", "Una tabla SQL", "Una imagen"], "Un número"),
        ("En este ejemplo, ¿cuál es el target?", ["rating_numerico", "titulo", "genero_principal"], "rating_numerico"),
        ("¿Qué significa un coeficiente positivo?", ["Que la variable tiende a subir la predicción", "Que la variable se elimina", "Que el modelo falló"], "Que la variable tiende a subir la predicción"),
        ("¿Qué mide MAE?", ["El error promedio absoluto", "La cantidad de filas duplicadas", "La cantidad de géneros"], "El error promedio absoluto"),
        ("¿Por qué usamos test?", ["Para evaluar con datos que el modelo no vio", "Para entrenar dos veces lo mismo", "Para borrar variables categóricas"], "Para evaluar con datos que el modelo no vio"),
        ("¿Qué pasa si agregamos variables de ruido?", ["Puede empeorar la generalización", "Siempre mejora", "Convierte la regresión en SQL"], "Puede empeorar la generalización"),
    ]
    score = 0
    for idx, (question, options, correct) in enumerate(quiz, start=1):
        answer = st.radio(f"Pregunta {idx}: {question}", options, key=f"movie_quiz_{idx}")
        if answer == correct:
            score += 1
    st.metric("Puntaje final", f"{score}/{len(quiz)}")


def render_movie_regression_section_previous_sklearn_demo() -> None:
    st.header("Regresión lineal para predecir rating de películas")
    df_movies, movie_source = cached_movie_data(900, 42)
    st.caption(movie_source)

    st.markdown(
        """
        <div class="concept-box">
        Una regresión lineal intenta predecir un número. En este caso, queremos predecir el rating numérico de una película
        usando características como duración, presupuesto, popularidad, reseñas positivas y experiencia del director.
        <br><br>
        El modelo no entiende la película como una persona. Solo ve números. Aprende qué variables tienden a subir o bajar el rating.
        </div>
        """,
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns([1, 1, 1])
    f1.markdown('<div class="concept-box"><b>Características de la película</b><br>género, duración, popularidad, reseñas</div>', unsafe_allow_html=True)
    f2.markdown('<div class="concept-box"><b>Regresión Lineal</b><br>rating = bias + pesos * variables</div>', unsafe_allow_html=True)
    f3.markdown('<div class="concept-box"><b>Rating estimado</b><br>Ejemplo: rating_numerico = 8.2</div>', unsafe_allow_html=True)
    st.code("Simple: y = m*x + b\nMultiple: rating = b + w1*x1 + w2*x2 + w3*x3 + ...")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Películas", f"{len(df_movies):,}")
    c2.metric("Rating promedio", f"{df_movies[MOVIE_TARGET].mean():.2f}")
    c3.metric("Features numéricas", len(MOVIE_NUMERIC_FEATURES))
    c4.metric("Feature categórica", ", ".join(MOVIE_CATEGORICAL_FEATURES))

    section("1. Primero: una sola variable")
    simple_feature = st.selectbox("Variable para explicar una regresión simple", SIMPLE_REGRESSION_OPTIONS)
    simple_model, simple_data, simple_metrics = simple_regression_fit(df_movies, simple_feature)
    slope = float(simple_model.coef_[0])
    intercept = float(simple_model.intercept_)
    left, right = st.columns([1.45, 1])
    with left:
        st.plotly_chart(simple_regression_plot(simple_data, simple_feature, simple_model), width="stretch", key=f"simple_reg_{simple_feature}")
    with right:
        st.metric("Pendiente m", f"{slope:.4f}")
        st.metric("Intercepto b", f"{intercept:.3f}")
        st.code(f"rating = {slope:.4f} * {simple_feature} + {intercept:.3f}")
        st.write(
            "La línea intenta pasar lo más cerca posible de los puntos. Cada punto es una película. "
            "Si la pendiente es positiva, cuando la variable sube, el rating tiende a subir."
        )
        for name, value in simple_metrics.items():
            st.metric(name, f"{value:.3f}")
    questions(["¿La línea parece representar bien los puntos?", "¿La pendiente es positiva o negativa?", "¿La variable elegida parece tener relación con el rating?", "¿Hay puntos alejados de la línea?"])

    section("2. Error: qué intenta minimizar el modelo")
    st.plotly_chart(simple_regression_plot(simple_data, simple_feature, simple_model, show_errors=True), width="stretch", key=f"errors_{simple_feature}")
    example = simple_data.sample(1, random_state=9).iloc[0]
    real = float(example[MOVIE_TARGET])
    pred = float(example["prediccion"])
    st.write(
        f"El error es la diferencia entre el rating real y el rating predicho. "
        f"Ejemplo: rating real = {real:.1f}, rating predicho = {pred:.1f}, error = {abs(real - pred):.1f}."
    )
    st.write("La regresión lineal busca una línea que reduzca esos errores en conjunto.")
    e1, e2, e3 = st.columns(3)
    e1.info("MAE: error promedio absoluto. En promedio, cuántos puntos de rating se equivoca el modelo.")
    e2.info("RMSE: similar al MAE, pero castiga más los errores grandes.")
    e3.info("R2: qué tanto de la variación del rating logra explicar el modelo. Mientras más cerca de 1, mejor.")

    section("3. Ahora: varias variables")
    st.write("Cuando usamos varias variables, el modelo ya no aprende una sola pendiente. Aprende un peso para cada variable.")
    st.code("rating = bias\n       + peso_1 * reseñas_positivas_pct\n       + peso_2 * popularidad\n       + peso_3 * presupuesto_millones\n       + ...")
    config_left, config_right = st.columns([1.2, 1])
    with config_left:
        st.write("**Features numéricas**")
        selected_numeric = [feature for feature in MOVIE_NUMERIC_FEATURES if st.checkbox(feature, value=True, key=f"movie_feature_{feature}")]
        use_genre = st.checkbox("genero_principal", value=True, key="movie_feature_genero")
    with config_right:
        test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05, key="movie_test_size")
        random_state = st.number_input("random_state", value=42, step=1, key="movie_random_state")
        use_scaling = st.checkbox("Usar normalización", value=True, key="movie_scaling")
        st.plotly_chart(split_bar((1 - test_size) * 100, test_size * 100), width="stretch", key="movie_split_bar")
        st.write("El modelo aprende con train y se evalúa con test. Test simula películas nuevas que el modelo no vio.")
    questions(["¿Por qué no evaluamos solo con los datos de entrenamiento?", "¿Qué pasaría si el modelo memoriza?", "¿Qué representa test en la vida real?"])

    selected_features = selected_numeric + (["genero_principal"] if use_genre else [])
    if not selected_features:
        st.warning("Selecciona al menos una variable para entrenar.")
        return

    section("4. Normalización")
    norm_cols = ["presupuesto_millones", "votos_usuarios", "reseñas_positivas_pct", "duracion_min"]
    st.dataframe(normalization_comparison_table(df_movies, norm_cols), width="stretch")
    st.write(
        "Normalizar pone las variables en escalas comparables. Después de normalizar, quedan con media cercana a 0 "
        "y desviación estándar cercana a 1."
    )
    questions(["¿Qué variable tenía la escala más grande antes?", "¿Por qué puede ser difícil comparar coeficientes sin normalizar?", "¿Cambió mucho el desempeño?"])

    section("5. Entrenar modelo")
    train_clicked = st.button("Entrenar regresión lineal", type="primary")
    state_key = "movie_regression_result"
    if train_clicked:
        st.session_state[state_key] = cached_movie_train(
            df_movies,
            tuple(selected_features),
            bool(use_scaling),
            float(test_size),
            int(random_state),
        )
        st.session_state["movie_regression_config"] = {
            "features": selected_features,
            "use_scaling": bool(use_scaling),
            "test_size": float(test_size),
            "random_state": int(random_state),
        }
    result = st.session_state.get(state_key)

    if result is None:
        st.info("Presiona el botón para entrenar el modelo múltiple.")
    else:
        trained_config = st.session_state.get(
            "movie_regression_config",
            {"features": selected_features, "use_scaling": bool(use_scaling)},
        )
        trained_features = trained_config["features"]
        if trained_features != selected_features:
            st.info("Cambiaste la selección de variables. Presiona de nuevo el botón de entrenamiento para actualizar el modelo.")
        train_test_metrics_row(result.metrics)
        g1, g2 = st.columns([1.1, 1])
        with g1:
            st.write("**Rating real vs rating predicho**")
            st.plotly_chart(rating_real_vs_pred_plot(result.y_test, result.y_pred_test), width="stretch", key="movie_real_vs_pred")
        hist, resid = residual_plots(result.y_test, result.y_pred_test)
        with g2:
            st.write("**Histograma de errores**")
            st.plotly_chart(hist, width="stretch", key="movie_error_hist")
        st.write("**Residuos vs predicción**")
        st.plotly_chart(resid, width="stretch", key="movie_residuals")
        st.write("Si las predicciones fueran perfectas, los puntos caerían cerca de la línea diagonal.")
        questions(["¿El modelo predice mejor ratings altos o bajos?", "¿Hay muchos errores grandes?", "¿Los errores parecen aleatorios o tienen patrón?", "¿Train es mucho mejor que test?"])

        section("6. Coeficientes del modelo")
        coef_table = result.coefficients.drop(columns=["impacto_abs"]).copy()
        st.dataframe(coef_table, width="stretch", height=360)
        st.write(
            "Un coeficiente positivo significa que, manteniendo lo demás constante, esa variable tiende a subir el rating. "
            "Un coeficiente negativo significa que tiende a bajarlo."
        )
        if trained_config.get("use_scaling"):
            st.info("Con normalización, los coeficientes son más comparables entre variables.")
        else:
            st.warning("Sin normalización, los coeficientes pueden ser difíciles de comparar porque las variables tienen escalas distintas.")
        questions(["¿Qué variables suben más el rating?", "¿Qué variables parecen bajar el rating?", "¿Tiene sentido que reseñas_positivas_pct tenga coeficiente positivo?", "¿Presupuesto alto siempre significa mejor rating?"])

        section("7. Predice el rating de tu propia película")
        genres = sorted(df_movies["genero_principal"].dropna().unique().tolist())
        with st.form("movie_manual_prediction"):
            p1, p2, p3 = st.columns(3)
            manual = {
                "genero_principal": p1.selectbox("genero_principal", genres, index=genres.index("Drama") if "Drama" in genres else 0),
                "duracion_min": p1.slider("duracion_min", 70, 190, 125),
                "presupuesto_millones": p1.slider("presupuesto_millones", 1.0, 260.0, 45.0),
                "marketing_millones": p1.slider("marketing_millones", 0.5, 160.0, 20.0),
                "votos_usuarios": p2.number_input("votos_usuarios", min_value=500, max_value=850000, value=80000, step=1000),
                "popularidad": p2.slider("popularidad", 0.0, 100.0, 76.0),
                "reseñas_positivas_pct": p2.slider("reseñas_positivas_pct", 0.0, 100.0, 88.0),
                "director_experiencia_anios": p2.slider("director_experiencia_anios", 0, 40, 14),
                "actores_populares": p3.slider("actores_populares", 0, 5, 2),
                "premios_previos": p3.slider("premios_previos", 0, 12, 2),
                "es_franquicia": p3.selectbox("es_franquicia", [0, 1]),
                "edad_recomendada": p3.selectbox("edad_recomendada", [0, 7, 12, 13, 16, 18], index=3),
                "nivel_accion": p1.slider("nivel_accion", 0.0, 10.0, 3.0),
                "nivel_comedia": p1.slider("nivel_comedia", 0.0, 10.0, 2.0),
                "nivel_drama": p2.slider("nivel_drama", 0.0, 10.0, 8.0),
                "nivel_romance": p3.slider("nivel_romance", 0.0, 10.0, 3.0),
                "nivel_suspenso": p3.slider("nivel_suspenso", 0.0, 10.0, 4.0),
            }
            submitted = st.form_submit_button("Predecir rating")
        if submitted:
            row = pd.DataFrame([{feature: manual[feature] for feature in trained_features}])
            prediction = float(np.clip(result.pipeline.predict(row)[0], 1.0, 10.0))
            category = "Bajo" if prediction < 6.0 else "Medio" if prediction < 7.2 else "Alto" if prediction < 8.3 else "Excelente"
            m1, m2, m3 = st.columns(3)
            m1.metric("Rating predicho", f"{prediction:.2f}")
            m2.metric("Rating redondeado", f"{round(prediction, 1):.1f}")
            m3.metric("Categoría interpretativa", category)
            st.write(
                "Esta categoría es solo una interpretación posterior. El modelo principal sigue siendo regresión lineal porque predice un número. "
                f"El modelo estima que esta película tendría rating {prediction:.2f} porque sus características se parecen a películas históricas con ratings similares."
            )

    section("8. Experimento: agregar o quitar variables")
    comparison = movie_model_comparison(df_movies, float(test_size), int(random_state), bool(use_scaling))
    st.dataframe(comparison, width="stretch")
    questions(["¿Agregar más variables mejoró el modelo?", "¿La mejora fue grande o pequeña?", "¿Hay variables que agregan ruido?", "¿Cuál modelo escogerías y por qué?"])

    section("9. Experimento: variables correlacionadas")
    add_dups = st.checkbox("Agregar variables duplicadas/correlacionadas", key="movie_add_dups")
    df_corr = add_correlated_movie_features(df_movies, int(random_state)) if add_dups else df_movies.copy()
    if add_dups:
        st.plotly_chart(px.imshow(df_corr.select_dtypes(include=[np.number]).corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"), width="stretch", key="movie_corr_heatmap")
        pairs = high_movie_correlations(df_corr, 0.85)
        st.dataframe(pairs if not pairs.empty else pd.DataFrame({"mensaje": ["No hay pares sobre 0.85."]}), width="stretch")
    base = train_movie_regression(df_movies, MOVIE_FEATURES, bool(use_scaling), float(test_size), int(random_state))
    dup = train_movie_regression(add_correlated_movie_features(df_movies, int(random_state)), MOVIE_FEATURES + ["popularidad_dup", "reseñas_positivas_dup", "marketing_dup"], bool(use_scaling), float(test_size), int(random_state))
    st.dataframe(
        pd.DataFrame(
            [
                {"Modelo": "Sin variables duplicadas", "MAE test": base.metrics["mae_test"], "RMSE test": base.metrics["rmse_test"], "R2 test": base.metrics["r2_test"]},
                {"Modelo": "Con variables duplicadas", "MAE test": dup.metrics["mae_test"], "RMSE test": dup.metrics["rmse_test"], "R2 test": dup.metrics["r2_test"]},
            ]
        ),
        width="stretch",
    )
    st.write("Variables muy correlacionadas pueden no aportar información nueva y pueden hacer que los coeficientes sean inestables o difíciles de interpretar.")
    questions(["¿Mejoró el MAE?", "¿Mejoró el R2?", "¿Los coeficientes cambiaron mucho?", "¿Agregar columnas repetidas hizo el modelo más claro o más confuso?"])

    section("10. Overfitting en regresión lineal")
    st.write(
        "La regresión lineal simple no suele memorizar tanto como modelos muy flexibles, pero puede sobreajustarse si agregamos demasiadas variables, "
        "transformaciones innecesarias o ruido."
    )
    add_noise = st.button("Agregar variables de ruido")
    overfit_df = overfitting_regression_comparison(df_movies, float(test_size), int(random_state), bool(use_scaling), add_noise)
    st.dataframe(overfit_df, width="stretch")
    st.write("Si train mejora pero test no mejora o empeora, el modelo está aprendiendo cosas que no generalizan.")
    questions(["¿Train mejoró al agregar ruido?", "¿Test mejoró realmente?", "¿Qué modelo generaliza mejor?"])

    section("11. Mini quiz")
    render_movie_quiz()

    section("12. Retos finales")
    retos = [
        ("Reto 1", "Usa solo reseñas_positivas_pct. ¿Qué MAE obtuviste?"),
        ("Reto 2", "Agrega popularidad y votos_usuarios. ¿Mejoró el modelo?"),
        ("Reto 3", "Activa normalización. ¿Cambió el desempeño o principalmente la interpretación?"),
        ("Reto 4", "Agrega variables correlacionadas. ¿Mejoró o solo complicó los coeficientes?"),
        ("Reto 5", "Agrega variables de ruido. ¿Train mejora más que test?"),
        ("Reto 6", "Crea una película manualmente. ¿Qué variables tuviste que subir para obtener rating alto?"),
    ]
    cols = st.columns(2)
    for idx, (title, text) in enumerate(retos):
        with cols[idx % 2]:
            done = st.checkbox(title, key=f"movie_challenge_{idx}")
            st.caption(text if not done else f"Completado. {text}")


def manual_line_plot(df_manual: pd.DataFrame, m: float, b: float, title: str) -> go.Figure:
    plot_df = df_manual.copy()
    plot_df["rating_predicho"] = predict_simple(plot_df[SIMPLE_FEATURE], m, b)
    x_line = np.linspace(plot_df[SIMPLE_FEATURE].min(), plot_df[SIMPLE_FEATURE].max(), 120)
    y_line = predict_simple(x_line, m, b)
    fig = px.scatter(
        plot_df,
        x=SIMPLE_FEATURE,
        y=MANUAL_MOVIE_TARGET,
        text="titulo" if "titulo" in plot_df.columns else None,
        color_discrete_sequence=["#457b9d"],
        title=title,
    )
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Modelo actual", line=dict(color="#e76f51", width=3)))
    error_rows = plot_df if len(plot_df) <= 80 else plot_df.sample(80, random_state=7)
    for _, row in error_rows.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[SIMPLE_FEATURE], row[SIMPLE_FEATURE]],
                y=[row[MANUAL_MOVIE_TARGET], row["rating_predicho"]],
                mode="lines",
                showlegend=False,
                line=dict(color="#111827", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=45, b=20), yaxis_title="rating_real")
    return fig


def history_line(history: list[dict], y_col: str, title: str) -> go.Figure:
    hist = pd.DataFrame(history)
    if hist.empty:
        hist = pd.DataFrame({"iteracion": [0], y_col: [0]})
    fig = px.line(hist, x="iteracion", y=y_col, markers=True, title=title, color_discrete_sequence=["#2a9d8f"])
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def cost_function_plot(x, y, best_m: float, b: float) -> go.Figure:
    m_values = np.linspace(best_m - 0.08, best_m + 0.08, 120)
    cost_rows = []
    for m_value in m_values:
        y_pred = predict_simple(x, m_value, b)
        cost_rows.append({"m": m_value, "MSE": compute_mse(y, y_pred)})
    cost_df = pd.DataFrame(cost_rows)
    best_cost = compute_mse(y, predict_simple(x, best_m, b))
    fig = px.line(cost_df, x="m", y="MSE", title="Función de coste: MSE al cambiar la pendiente m", color_discrete_sequence=["#457b9d"])
    fig.add_trace(
        go.Scatter(
            x=[best_m],
            y=[best_cost],
            mode="markers",
            name="m calculada",
            marker=dict(color="#e76f51", size=11),
        )
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def manual_quiz() -> None:
    quiz = [
        ("¿Qué intenta predecir la regresión lineal?", ["Un número", "Una categoría", "Un archivo SQL"], "Un número"),
        ("¿Qué representa la pendiente m?", ["Cuánto cambia la predicción cuando cambia x", "El nombre de la película", "La cantidad de columnas"], "Cuánto cambia la predicción cuando cambia x"),
        ("¿Qué representa el error?", ["La diferencia entre lo real y lo predicho", "El número de filas", "El nombre de la variable"], "La diferencia entre lo real y lo predicho"),
        ("¿Qué hace el entrenamiento?", ["Ajusta parámetros para reducir el error", "Borra el dataset", "Cambia los títulos"], "Ajusta parámetros para reducir el error"),
        ("¿Por qué la computadora puede entrenar con miles de datos?", ["Porque repite operaciones matemáticas muchas veces", "Porque adivina", "Porque no necesita datos"], "Porque repite operaciones matemáticas muchas veces"),
        ("¿Por qué normalizamos?", ["Para poner variables en escalas comparables", "Para eliminar el target", "Para convertir todo en texto"], "Para poner variables en escalas comparables"),
    ]
    score = 0
    for idx, (prompt, options, correct) in enumerate(quiz, start=1):
        answer = st.radio(f"Pregunta {idx}: {prompt}", options, key=f"manual_lr_quiz_{idx}")
        score += int(answer == correct)
    st.metric("Puntaje final", f"{score}/{len(quiz)}")


def render_movie_regression_section_previous_gradient_demo() -> None:
    st.header("Regresión lineal hecha a mano")
    st.markdown(
        """
        <div class="concept-box">
        Este laboratorio muestra que un modelo no es magia: es una fórmula que predice, mide su error y ajusta sus parámetros
        muchas veces. Usaremos películas para predecir un rating numérico a partir del porcentaje de reseñas positivas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_small = small_manual_movie_dataset()
    x_small = df_small[SIMPLE_FEATURE].to_numpy(dtype=float)
    y_small = df_small[MANUAL_MOVIE_TARGET].to_numpy(dtype=float)

    section("1. Dataset pequeño para trabajar a mano")
    st.write("Para empezar usaremos una sola variable: porcentaje de reseñas positivas. Queremos predecir el rating de la película.")
    st.dataframe(df_small, width="stretch", hide_index=True)

    section("2. Fórmula de la regresión lineal simple")
    c1, c2 = st.columns([1, 1])
    c1.markdown(
        """
        <div class="concept-box">
        <b>rating_predicho = m * reseñas_positivas_pct + b</b>
        <br><br>
        m = pendiente<br>
        b = intercepto<br>
        x = reseñas positivas<br>
        y = rating real<br>
        y_pred = rating predicho
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.write(
        "La pendiente indica cuánto cambia el rating cuando suben las reseñas positivas. "
        "El intercepto es el punto de inicio de la línea."
    )

    section("3. Modo manual: mover la línea a mano")
    m_manual = st.slider("pendiente m", -0.2, 0.2, 0.03, 0.001, key="manual_slider_m")
    b_manual = st.slider("intercepto b", 0.0, 10.0, 3.0, 0.1, key="manual_slider_b")
    pred_manual = predict_simple(x_small, m_manual, b_manual)
    errors_manual = compute_errors(y_small, pred_manual)
    manual_table = pd.concat([df_small, errors_manual], axis=1)
    metrics_manual = simple_metrics(y_small, pred_manual)
    st.dataframe(manual_table, width="stretch", hide_index=True)
    mcols = st.columns(3)
    mcols[0].metric("MAE", f"{metrics_manual['mae']:.3f}")
    mcols[1].metric("MSE", f"{metrics_manual['mse']:.3f}")
    mcols[2].metric("RMSE", f"{metrics_manual['rmse']:.3f}")
    st.write("MAE es el error promedio absoluto. MSE es el promedio de errores al cuadrado. RMSE es el error típico en unidades de rating.")

    section("4. Gráfica interactiva")
    st.plotly_chart(manual_line_plot(df_small, m_manual, b_manual, "Puntos reales, línea actual y errores"), width="stretch", key="manual_line_plot")
    st.write("La regresión lineal busca una línea que pase lo más cerca posible de los puntos.")

    section("5. Reto manual para la estudiante")
    r1, r2 = st.columns(2)
    r1.metric("MAE actual", f"{metrics_manual['mae']:.3f}")
    r2.metric("RMSE actual", f"{metrics_manual['rmse']:.3f}")
    if metrics_manual["rmse"] > 1.0:
        st.warning("Todavía hay bastante error.")
    elif metrics_manual["rmse"] >= 0.5:
        st.info("La línea ya se acerca bastante.")
    else:
        st.success("Muy buen ajuste.")
    questions(["¿Qué pasa si subís mucho la pendiente?", "¿Qué pasa si bajás mucho el intercepto?", "¿Podés encontrar una línea que reduzca el error?", "¿La línea perfecta existe o siempre queda algo de error?"])

    section("6. Una iteración como la computadora")
    if "manual_gd_m" not in st.session_state:
        st.session_state.manual_gd_m = 0.03
        st.session_state.manual_gd_b = 3.0
        st.session_state.manual_gd_history = []
    learning_rate = st.slider("learning_rate", 0.000001, 0.01, 0.0001, format="%.6f", key="manual_learning_rate")
    before_pred = predict_simple(x_small, st.session_state.manual_gd_m, st.session_state.manual_gd_b)
    dm, db = compute_gradients_simple(x_small, y_small, before_pred)
    next_m, next_b = update_simple_params(st.session_state.manual_gd_m, st.session_state.manual_gd_b, dm, db, learning_rate)
    after_pred = predict_simple(x_small, next_m, next_b)
    iter_table = pd.concat(
        [
            df_small[["titulo", SIMPLE_FEATURE, MANUAL_MOVIE_TARGET]],
            pd.DataFrame({"prediccion": before_pred, "error": before_pred - y_small}),
        ],
        axis=1,
    )
    st.dataframe(iter_table, width="stretch", hide_index=True)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("m actual", f"{st.session_state.manual_gd_m:.5f}")
    i2.metric("b actual", f"{st.session_state.manual_gd_b:.5f}")
    i3.metric("gradiente dm", f"{dm:.3f}")
    i4.metric("gradiente db", f"{db:.3f}")
    st.write(f"Error antes: {compute_mse(y_small, before_pred):.4f} · Error después de esta iteración: {compute_mse(y_small, after_pred):.4f}")
    if st.button("Siguiente iteración", type="primary"):
        new_m, new_b, hist, stable = train_simple_steps(
            x_small,
            y_small,
            st.session_state.manual_gd_m,
            st.session_state.manual_gd_b,
            learning_rate,
            1,
            st.session_state.manual_gd_history,
        )
        if stable:
            st.session_state.manual_gd_m = new_m
            st.session_state.manual_gd_b = new_b
            st.session_state.manual_gd_history = hist
            st.rerun()
        else:
            st.warning("El entrenamiento se volvió inestable. Bajá el learning rate o reiniciá el entrenamiento.")

    section("7. Entrenamiento automático")
    b1, b2, b3, b4 = st.columns(4)
    steps_to_run = 0
    if b1.button("Entrenar 10 iteraciones"):
        steps_to_run = 10
    if b2.button("Entrenar 100 iteraciones"):
        steps_to_run = 100
    if b3.button("Entrenar 1000 iteraciones"):
        steps_to_run = 1000
    if b4.button("Reiniciar entrenamiento"):
        st.session_state.manual_gd_m = 0.03
        st.session_state.manual_gd_b = 3.0
        st.session_state.manual_gd_history = []
        st.rerun()
    if steps_to_run:
        new_m, new_b, hist, stable = train_simple_steps(
            x_small,
            y_small,
            st.session_state.manual_gd_m,
            st.session_state.manual_gd_b,
            learning_rate,
            steps_to_run,
            st.session_state.manual_gd_history,
        )
        if stable:
            st.session_state.manual_gd_m = new_m
            st.session_state.manual_gd_b = new_b
            st.session_state.manual_gd_history = hist
            st.rerun()
        else:
            st.warning("El error explotó o se volvió infinito. Probá un learning rate más pequeño.")

    gd_pred = predict_simple(x_small, st.session_state.manual_gd_m, st.session_state.manual_gd_b)
    gd_metrics = simple_metrics(y_small, gd_pred)
    gcols = st.columns(5)
    gcols[0].metric("Iteración actual", len(st.session_state.manual_gd_history))
    gcols[1].metric("m actual", f"{st.session_state.manual_gd_m:.5f}")
    gcols[2].metric("b actual", f"{st.session_state.manual_gd_b:.5f}")
    gcols[3].metric("MSE actual", f"{gd_metrics['mse']:.4f}")
    gcols[4].metric("RMSE actual", f"{gd_metrics['rmse']:.4f}")
    st.write("Entrenar significa repetir muchas veces: predecir, medir error y ajustar m y b.")
    h1, h2 = st.columns(2)
    h1.plotly_chart(history_line(st.session_state.manual_gd_history, "mse", "MSE por iteración"), width="stretch", key="manual_mse_history")
    h2.plotly_chart(history_line(st.session_state.manual_gd_history, "rmse", "RMSE por iteración"), width="stretch", key="manual_rmse_history")
    h3, h4 = st.columns(2)
    h3.plotly_chart(history_line(st.session_state.manual_gd_history, "m", "Evolución de m"), width="stretch", key="manual_m_history")
    h4.plotly_chart(history_line(st.session_state.manual_gd_history, "b", "Evolución de b"), width="stretch", key="manual_b_history")

    section("8. Experimento con learning rate")
    st.write("Learning rate es el tamaño del paso que da el modelo cuando ajusta sus parámetros.")
    lr_cols = st.columns(3)
    lr_cols[0].info("Muy bajo: aprende muy lento.")
    lr_cols[1].success("Adecuado: baja el error de forma estable.")
    lr_cols[2].warning("Muy alto: puede saltarse la mejor solución y hacer que el error suba.")
    questions(["¿Qué pasa si el learning rate es muy pequeño?", "¿Qué pasa si es demasiado grande?", "¿Cuál parece funcionar mejor?"])

    section("9. De hacerlo a mano a hacerlo a gran escala")
    st.write(
        "Con 5 películas podemos calcular los errores a mano. Con 100,000 sería imposible. "
        "La computadora hace las mismas operaciones, pero miles o millones de veces."
    )
    if st.button("Generar dataset grande"):
        st.session_state.manual_large_df = generate_large_movie_dataset(1000, 42)
    df_large = st.session_state.get("manual_large_df")
    if df_large is not None:
        st.metric("Cantidad de filas", f"{len(df_large):,}")
        st.plotly_chart(
            manual_line_plot(df_large.assign(titulo=""), st.session_state.manual_gd_m, st.session_state.manual_gd_b, "Misma línea sobre 1000 películas"),
            width="stretch",
            key="large_simple_plot",
        )
        large_pred = predict_simple(df_large[SIMPLE_FEATURE], st.session_state.manual_gd_m, st.session_state.manual_gd_b)
        large_metrics = simple_metrics(df_large[MANUAL_MOVIE_TARGET], large_pred)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Dataset": "6 películas", "RMSE": gd_metrics["rmse"], "MSE": gd_metrics["mse"]},
                    {"Dataset": "1000 películas", "RMSE": large_metrics["rmse"], "MSE": large_metrics["mse"]},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    questions(["¿Por qué ya no conviene hacerlo a mano?", "¿Qué parte del proceso sigue siendo la misma?", "¿Qué ventaja tiene que la computadora repita miles de veces?"])

    section("10. Regresión lineal múltiple manual conceptual")
    df_multi = st.session_state.get("manual_large_df", generate_large_movie_dataset(1000, 42))
    X_norm, means, stds, norm_stats = normalize_features(df_multi, MULTIPLE_FEATURES)
    y_multi = df_multi[MANUAL_MOVIE_TARGET].to_numpy(dtype=float)
    if "manual_multi_weights" not in st.session_state:
        st.session_state.manual_multi_weights = np.zeros(len(MULTIPLE_FEATURES))
        st.session_state.manual_multi_bias = float(y_multi.mean())
        st.session_state.manual_multi_history = []
    st.code(
        "rating_predicho = b\n"
        "+ w1 * reseñas_positivas_pct\n"
        "+ w2 * popularidad\n"
        "+ w3 * presupuesto_millones\n"
        "+ w4 * experiencia_director"
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Variable": MULTIPLE_FEATURES,
                "Peso": [f"w{idx + 1}" for idx in range(len(MULTIPLE_FEATURES))],
                "Interpretación": ["cuánto empuja el rating"] * len(MULTIPLE_FEATURES),
            }
        ),
        width="stretch",
        hide_index=True,
    )
    multi_lr = st.slider("learning_rate múltiple", 0.000001, 0.1, 0.03, format="%.6f", key="manual_multi_lr")
    if st.button("Entrenar regresión múltiple"):
        weights, bias, hist, stable = train_multiple_steps(
            X_norm,
            y_multi,
            st.session_state.manual_multi_weights,
            st.session_state.manual_multi_bias,
            multi_lr,
            600,
            st.session_state.manual_multi_history,
        )
        if stable:
            st.session_state.manual_multi_weights = weights
            st.session_state.manual_multi_bias = bias
            st.session_state.manual_multi_history = hist
            st.rerun()
        else:
            st.warning("La regresión múltiple se volvió inestable. Bajá el learning rate.")
    multi_pred = predict_multiple(X_norm, st.session_state.manual_multi_weights, st.session_state.manual_multi_bias)
    multi_mse = compute_mse(y_multi, multi_pred)
    mc = st.columns(4)
    mc[0].metric("Bias", f"{st.session_state.manual_multi_bias:.4f}")
    mc[1].metric("MSE", f"{multi_mse:.4f}")
    mc[2].metric("RMSE", f"{np.sqrt(multi_mse):.4f}")
    mc[3].metric("Iteraciones", len(st.session_state.manual_multi_history))
    st.dataframe(
        pd.DataFrame({"Variable": MULTIPLE_FEATURES, "Peso actual": st.session_state.manual_multi_weights}),
        width="stretch",
        hide_index=True,
    )
    st.plotly_chart(history_line(st.session_state.manual_multi_history, "mse", "Historial de error múltiple"), width="stretch", key="multi_mse_history")
    st.write("La idea es la misma que con una variable. La diferencia es que ahora el modelo ajusta varios pesos al mismo tiempo.")

    section("11. Normalización visual")
    st.dataframe(
        pd.DataFrame(
            {
                "Variable": MULTIPLE_FEATURES,
                "Escala original": ["0 a 100", "0 a 100", "1 a 300", "0 a 40"],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.write("Si una variable tiene números mucho más grandes, puede afectar el entrenamiento. Normalizar ayuda a ponerlas en escalas comparables.")
    st.dataframe(norm_stats, width="stretch", hide_index=True)
    questions(["¿Qué variable tenía la escala más grande?", "¿Por qué presupuesto puede dominar si no se normaliza?", "¿Qué significa que después de normalizar la media sea cercana a 0?"])

    section("12. Comparación con scikit-learn")
    st.dataframe(sklearn_comparison(df_multi, multi_pred), width="stretch", hide_index=True)
    st.write(
        "Lo que hicimos manualmente es una versión educativa. scikit-learn hace operaciones parecidas, "
        "pero más optimizadas y listas para trabajar con muchos datos."
    )

    section("13. Mini quiz")
    manual_quiz()

    section("14. Retos finales")
    retos = [
        ("Reto 1", "Ajustá manualmente m y b hasta reducir el RMSE."),
        ("Reto 2", "Probá un learning rate muy pequeño. ¿Qué pasa con el error?"),
        ("Reto 3", "Probá un learning rate grande. ¿El error baja o se vuelve inestable?"),
        ("Reto 4", "Entrená con 5 películas y luego con 1000. ¿Qué cambia y qué se mantiene igual?"),
        ("Reto 5", "Entrená regresión múltiple. ¿Qué variable tiene el peso más fuerte?"),
        ("Reto 6", "Compará tu modelo manual con scikit-learn. ¿Qué tan cerca quedaron?"),
    ]
    reto_cols = st.columns(2)
    for idx, (title, text) in enumerate(retos):
        with reto_cols[idx % 2]:
            done = st.checkbox(title, key=f"manual_lr_reto_{idx}")
            st.caption(text if not done else f"Completado. {text}")


def closed_form_values(df_manual: pd.DataFrame) -> dict:
    x = df_manual[SIMPLE_FEATURE].to_numpy(dtype=float)
    y = df_manual[MANUAL_MOVIE_TARGET].to_numpy(dtype=float)
    n = len(df_manual)
    sum_x = float(x.sum())
    sum_y = float(y.sum())
    sum_xy = float((x * y).sum())
    sum_x2 = float((x**2).sum())
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x**2
    m = numerator / denominator
    x_mean = sum_x / n
    y_mean = sum_y / n
    b = y_mean - m * x_mean
    return {
        "n": n,
        "sum_x": sum_x,
        "sum_y": sum_y,
        "sum_xy": sum_xy,
        "sum_x2": sum_x2,
        "numerator": numerator,
        "denominator": denominator,
        "m": m,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "b": b,
    }


def simple_manual_table(df_manual: pd.DataFrame, include_xy=False, include_x2=False, include_pred=False, include_error=False, m=None, b=None) -> pd.DataFrame:
    out = df_manual[["pelicula", SIMPLE_FEATURE, MANUAL_MOVIE_TARGET]].rename(
        columns={SIMPLE_FEATURE: "x", MANUAL_MOVIE_TARGET: "y real"}
    )
    x = df_manual[SIMPLE_FEATURE].to_numpy(dtype=float)
    y = df_manual[MANUAL_MOVIE_TARGET].to_numpy(dtype=float)
    if include_xy:
        out["x*y"] = x * y
    if include_x2:
        out["x²"] = x**2
    if include_pred and m is not None and b is not None:
        pred = predict_simple(x, m, b)
        out["y predicho"] = pred
        if include_error:
            error = y - pred
            out["error"] = error
            out["|error|"] = np.abs(error)
            out["error²"] = error**2
    return out


def scaled_movie_dataset(n_rows: int, random_state: int = 42) -> pd.DataFrame:
    if n_rows == 10:
        return small_manual_movie_dataset()[[SIMPLE_FEATURE, "popularidad", "presupuesto_millones", MANUAL_MOVIE_TARGET]].copy()
    return generate_large_movie_dataset(n_rows, random_state)[[SIMPLE_FEATURE, "popularidad", "presupuesto_millones", MANUAL_MOVIE_TARGET]].copy()


def plot_closed_line(df_manual: pd.DataFrame, m: float, b: float, show_errors: bool, title: str) -> go.Figure:
    plot_df = df_manual.copy()
    plot_df["rating_predicho"] = predict_simple(plot_df[SIMPLE_FEATURE], m, b)
    x_line = np.linspace(plot_df[SIMPLE_FEATURE].min(), plot_df[SIMPLE_FEATURE].max(), 160)
    y_line = predict_simple(x_line, m, b)
    hover = "pelicula" if "pelicula" in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x=SIMPLE_FEATURE,
        y=MANUAL_MOVIE_TARGET,
        hover_name=hover,
        opacity=0.75,
        title=title,
        color_discrete_sequence=["#457b9d"],
    )
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="línea del modelo", line=dict(color="#e76f51", width=3)))
    if show_errors:
        rows = plot_df if len(plot_df) <= 80 else plot_df.sample(80, random_state=7)
        for _, row in rows.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row[SIMPLE_FEATURE], row[SIMPLE_FEATURE]],
                    y=[row[MANUAL_MOVIE_TARGET], row["rating_predicho"]],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="#111827", dash="dot", width=1),
                    hoverinfo="skip",
                )
            )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=55, b=20), yaxis_title="rating_real")
    return fig


def metric_cards(values: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(values))
    for col, (label, value, help_text) in zip(cols, values):
        col.metric(label, value, help=help_text)


def render_guided_quiz() -> None:
    quiz = [
        ("¿Qué representa x?", ["La variable de entrada", "El error", "El intercepto"], "La variable de entrada"),
        ("¿Qué representa y?", ["El valor real que queremos predecir", "La pendiente", "El número de filas"], "El valor real que queremos predecir"),
        ("¿Qué representa m?", ["La pendiente de la línea", "El nombre de la película", "La suma de errores"], "La pendiente de la línea"),
        ("¿Qué representa b?", ["El intercepto", "La variable objetivo", "La cantidad de datos"], "El intercepto"),
        ("¿Qué es el error?", ["La diferencia entre lo real y lo predicho", "El número de variables", "La suma de x"], "La diferencia entre lo real y lo predicho"),
        ("¿Qué intenta hacer el entrenamiento?", ["Reducir el error ajustando parámetros", "Cambiar los títulos de las películas", "Eliminar la columna y"], "Reducir el error ajustando parámetros"),
        ("¿Por qué usamos la computadora con muchos datos?", ["Porque repite operaciones matemáticas muchas veces y rápido", "Porque adivina", "Porque no necesita datos"], "Porque repite operaciones matemáticas muchas veces y rápido"),
    ]
    score = 0
    for idx, (prompt, options, correct) in enumerate(quiz, start=1):
        answer = st.radio(f"Pregunta {idx}: {prompt}", options, key=f"guided_lr_quiz_{idx}")
        score += int(answer == correct)
    st.metric("Puntaje final", f"{score}/{len(quiz)}")


def render_movie_regression_section() -> None:
    st.header("Regresión lineal manual: de la tabla al modelo")
    st.info(
        "Primero lo hacemos a mano con pocos datos. Después vemos que la computadora hace el mismo tipo de operaciones, "
        "pero miles o millones de veces."
    )

    st.session_state.guided_lr_step = 10
    reveal_flags = [
        "guided_lr_show_xy",
        "guided_lr_show_x2",
        "guided_lr_show_sums",
        "guided_lr_show_m",
        "guided_lr_show_xmean",
        "guided_lr_show_ymean",
        "guided_lr_show_b",
        "guided_lr_show_pred",
        "guided_lr_show_error",
        "guided_lr_show_abs",
        "guided_lr_show_sq",
        "guided_lr_show_metrics",
    ]
    for flag in reveal_flags:
        st.session_state.setdefault(flag, False)
    if st.button("Mostrar todos los cálculos", key="manual_lr_show_all"):
        for flag in reveal_flags:
            st.session_state[flag] = True
        st.rerun()
    if st.button("Reiniciar cálculos interactivos", key="manual_lr_reset_all"):
        for key in list(st.session_state.keys()):
            if key.startswith("guided_lr_") or key.startswith("guided_multi_"):
                del st.session_state[key]
        st.rerun()

    df_small = small_manual_movie_dataset()
    vals = closed_form_values(df_small)
    x = df_small[SIMPLE_FEATURE].to_numpy(dtype=float)
    y = df_small[MANUAL_MOVIE_TARGET].to_numpy(dtype=float)

    if st.session_state.guided_lr_step >= 1:
        section("Paso 1. Ver datos")
        st.write("Cada fila es una película. Queremos encontrar una fórmula que use las reseñas positivas para estimar el rating.")
        st.dataframe(df_small, width="stretch", hide_index=True)
        questions(["¿Parece que cuando suben las reseñas también sube el rating?", "¿Crees que se puede dibujar una línea que aproxime estos datos?", "¿Crees que la línea será perfecta?"])

    if st.session_state.guided_lr_step >= 2:
        section("Paso 2. Calcular columnas auxiliares")
        st.write("Para la regresión simple usamos `x`, `y`, `x*y` y `x²`.")
        if st.button("Calcular x*y", key="guided_lr_btn_xy"):
            st.session_state.guided_lr_show_xy = True
        if st.button("Calcular x²", key="guided_lr_btn_x2"):
            st.session_state.guided_lr_show_x2 = True
        st.dataframe(
            simple_manual_table(
                df_small,
                include_xy=st.session_state.get("guided_lr_show_xy", False),
                include_x2=st.session_state.get("guided_lr_show_x2", False),
            ),
            width="stretch",
            hide_index=True,
        )
        st.write("Estas columnas auxiliares nos sirven para calcular la línea que mejor se ajusta a los datos.")
        questions(["¿Qué significa x*y?", "¿Qué significa x²?", "¿Por qué necesitamos sumar esas columnas?"])

    if st.session_state.guided_lr_step >= 3:
        section("Paso 3. Calcular sumatorias")
        if st.button("Calcular sumatorias", key="guided_lr_btn_sums"):
            st.session_state.guided_lr_show_sums = True
        if st.session_state.get("guided_lr_show_sums", False):
            metric_cards(
                [
                    ("n", f"{vals['n']}", "cantidad de datos"),
                    ("Σx", f"{vals['sum_x']:.1f}", "suma de reseñas"),
                    ("Σy", f"{vals['sum_y']:.1f}", "suma de ratings"),
                    ("Σxy", f"{vals['sum_xy']:.1f}", "suma de x*y"),
                    ("Σx²", f"{vals['sum_x2']:.1f}", "suma de x²"),
                ]
            )
        st.write("Las sumatorias resumen toda la tabla en pocos números. Con esos números calculamos la pendiente y el intercepto.")

    if st.session_state.guided_lr_step >= 4:
        section("Paso 4. Calcular pendiente e intercepto")
        st.latex(r"m = \frac{n\Sigma xy - \Sigma x \Sigma y}{n\Sigma x^2 - (\Sigma x)^2}")
        if st.button("Calcular pendiente m", key="guided_lr_btn_m"):
            st.session_state.guided_lr_show_m = True
        if st.session_state.get("guided_lr_show_m", False):
            st.write(f"Numerador = n * suma_xy - suma_x * suma_y = {vals['numerator']:.4f}")
            st.write(f"Denominador = n * suma_x2 - suma_x² = {vals['denominator']:.4f}")
            st.metric("Pendiente m", f"{vals['m']:.5f}")
        st.latex(r"b = \bar{y} - m\bar{x}")
        c1, c2, c3 = st.columns(3)
        if c1.button("Calcular x promedio", key="guided_lr_btn_xmean"):
            st.session_state.guided_lr_show_xmean = True
        if c2.button("Calcular y promedio", key="guided_lr_btn_ymean"):
            st.session_state.guided_lr_show_ymean = True
        if c3.button("Calcular intercepto b", key="guided_lr_btn_b"):
            st.session_state.guided_lr_show_b = True
        mparts = []
        if st.session_state.get("guided_lr_show_xmean", False):
            mparts.append(("x̄", f"{vals['x_mean']:.3f}", "Σx / n"))
        if st.session_state.get("guided_lr_show_ymean", False):
            mparts.append(("ȳ", f"{vals['y_mean']:.3f}", "Σy / n"))
        if st.session_state.get("guided_lr_show_b", False):
            mparts.append(("b", f"{vals['b']:.3f}", "intercepto"))
        if mparts:
            metric_cards(mparts)
        st.write("La pendiente indica cuánto cambia el rating cuando suben las reseñas positivas en una unidad.")
        questions(["Si m es positiva, ¿qué significa?", "Si m fuera negativa, ¿qué significaría?"])

    if st.session_state.guided_lr_step >= 5:
        section("Paso 5. Dibujar la línea")
        st.metric("Modelo final", f"rating_predicho = {vals['m']:.5f} * reseñas_positivas_pct + {vals['b']:.3f}")
        st.write("Esta es nuestra primera versión del modelo: una fórmula matemática que convierte reseñas positivas en rating estimado.")
        st.plotly_chart(plot_closed_line(df_small, vals["m"], vals["b"], False, "Ajuste de una línea recta a los datos"), width="stretch", key="guided_line_closed")
        questions(["¿La línea pasa exactamente por todos los puntos?", "¿Qué puntos están más lejos de la línea?", "¿La línea parece representar la tendencia general?"])

    if st.session_state.guided_lr_step >= 6:
        section("Paso 6. Calcular predicciones")
        st.write("Después de calcular `y predicho`, medimos qué tan lejos quedó cada predicción del valor real.")
        st.latex(r"error = y_{real} - y_{predicho}")
        st.latex(r"|error| = valor\ absoluto\ del\ error")
        st.latex(r"error^2 = error \times error")
        col_a, col_b, col_c, col_d = st.columns(4)
        if col_a.button("Revelar predicción", key="guided_lr_btn_pred"):
            st.session_state.guided_lr_show_pred = True
        if col_b.button("Revelar error", key="guided_lr_btn_error"):
            st.session_state.guided_lr_show_error = True
        if col_c.button("Revelar error absoluto", key="guided_lr_btn_abs"):
            st.session_state.guided_lr_show_abs = True
        if col_d.button("Revelar error cuadrado", key="guided_lr_btn_sq"):
            st.session_state.guided_lr_show_sq = True
        pred = predict_simple(x, vals["m"], vals["b"])
        table = simple_manual_table(df_small, include_pred=st.session_state.get("guided_lr_show_pred", False), m=vals["m"], b=vals["b"])
        if st.session_state.get("guided_lr_show_error", False) and "y predicho" in table:
            table["error"] = y - pred
        if st.session_state.get("guided_lr_show_abs", False) and "y predicho" in table:
            table["|error|"] = np.abs(y - pred)
        if st.session_state.get("guided_lr_show_sq", False) and "y predicho" in table:
            table["error²"] = (y - pred) ** 2
        st.dataframe(table, width="stretch", hide_index=True)
        st.write("El error nos dice qué tanto se equivocó el modelo en cada película.")

    if st.session_state.guided_lr_step >= 7:
        section("Paso 7. Calcular errores")
        st.write("Estas métricas resumen todos los errores individuales en pocos números para entender el error general del modelo.")
        st.latex(r"MAE = promedio(|error|)")
        st.write("MAE mide cuánto se equivoca el modelo en promedio, sin importar si el error fue positivo o negativo.")
        st.latex(r"MSE = promedio(error^2)")
        st.write("MSE eleva los errores al cuadrado, por eso castiga más los errores grandes.")
        st.latex(r"RMSE = \sqrt{MSE}")
        st.write("RMSE vuelve el error a una escala parecida al rating, por eso suele ser más fácil de interpretar que MSE.")
        if st.button("Calcular métricas de error", key="guided_lr_btn_metrics"):
            st.session_state.guided_lr_show_metrics = True
        pred = predict_simple(x, vals["m"], vals["b"])
        metrics = simple_metrics(y, pred)
        if st.session_state.get("guided_lr_show_metrics", False):
            metric_cards(
                [
                    ("MAE", f"{metrics['mae']:.3f}", "En promedio, cuántos puntos se equivoca."),
                    ("MSE", f"{metrics['mse']:.3f}", "Promedio de errores al cuadrado."),
                    ("RMSE", f"{metrics['rmse']:.3f}", "Error típico en unidades de rating."),
                ]
            )
        st.plotly_chart(plot_closed_line(df_small, vals["m"], vals["b"], True, "Errores como distancias verticales"), width="stretch", key="guided_error_lines")
        st.write("La función de coste muestra qué tan grande es el error total del modelo. Aquí usamos MSE como coste.")
        st.plotly_chart(cost_function_plot(x, y, vals["m"], vals["b"]), width="stretch", key="guided_cost_function")
        st.write("Cada línea vertical muestra cuánto se equivocó el modelo para esa película.")
        questions(["¿Cuál métrica es más fácil de interpretar?", "Si RMSE = 0.4, ¿qué significa?", "¿Un error de 0 sería realista?", "¿Qué película tiene mayor error?"])

    if st.session_state.guided_lr_step >= 8:
        section("Paso 8. Ajustar manualmente")
        st.write("Ahora intentá ajustar la línea manualmente para reducir el RMSE.")
        manual_m = st.slider("pendiente m", -0.2, 0.2, float(vals["m"]), 0.001, key="guided_manual_m")
        manual_b = st.slider("intercepto b", 0.0, 10.0, float(vals["b"]), 0.05, key="guided_manual_b")
        manual_pred = predict_simple(x, manual_m, manual_b)
        manual_metrics = simple_metrics(y, manual_pred)
        metric_cards(
            [
                ("MAE", f"{manual_metrics['mae']:.3f}", None),
                ("MSE", f"{manual_metrics['mse']:.3f}", None),
                ("RMSE", f"{manual_metrics['rmse']:.3f}", None),
            ]
        )
        if manual_metrics["rmse"] > 0.8:
            st.warning("La línea todavía está lejos.")
        elif manual_metrics["rmse"] > 0.35:
            st.info("Vas mejorando.")
        else:
            st.success("Buen ajuste.")
        st.plotly_chart(plot_closed_line(df_small, manual_m, manual_b, True, "Ajuste manual de m y b"), width="stretch", key="guided_manual_plot")
        questions(["¿Qué pasa si subís demasiado m?", "¿Qué pasa si bajás mucho b?", "¿Podés reducir el error manualmente?", "¿Por qué hacerlo a mano se vuelve difícil?"])

    if st.session_state.guided_lr_step >= 9:
        section("Paso 9. Entrenar automáticamente")
        st.write("La computadora también puede aprender ajustando m y b poco a poco. A esto se le llama gradient descent.")
        st.info(
            "Idea clave: en entrenamiento automático, la computadora calcula dos direcciones de ajuste: `dm` para la pendiente "
            "y `db` para el intercepto. Si `dm` o `db` son grandes, significa que el error empuja fuerte al modelo para cambiar ese parámetro."
        )
        st.write(
            "`learning_rate` es el tamaño del paso. Un valor pequeño cambia `m` y `b` lentamente; un valor grande cambia más rápido, "
            "pero puede pasarse de la mejor línea y volver inestable el error."
        )
        st.latex(r"y_{pred}=m x+b")
        st.latex(r"dm=promedio(2 \cdot error \cdot x), \quad db=promedio(2 \cdot error)")
        st.latex(r"m_{nuevo}=m-learning\_rate \cdot dm")
        st.latex(r"b_{nuevo}=b-learning\_rate \cdot db")
        if "guided_lr_gd_m" not in st.session_state:
            st.session_state.guided_lr_gd_m = 0.03
            st.session_state.guided_lr_gd_b = 3.0
            st.session_state.guided_lr_gd_history = []
        lr = st.slider("learning_rate", 0.000001, 0.01, 0.0001, format="%.6f", key="guided_lr_rate")
        current_pred = predict_simple(x, st.session_state.guided_lr_gd_m, st.session_state.guided_lr_gd_b)
        dm, db = compute_gradients_simple(x, y, current_pred)
        new_m, new_b = update_simple_params(st.session_state.guided_lr_gd_m, st.session_state.guided_lr_gd_b, dm, db, lr)
        new_pred = predict_simple(x, new_m, new_b)
        iter_row = pd.DataFrame(
            [
                {
                    "iteración": len(st.session_state.guided_lr_gd_history) + 1,
                    "m anterior": st.session_state.guided_lr_gd_m,
                    "b anterior": st.session_state.guided_lr_gd_b,
                    "dm": dm,
                    "db": db,
                    "m nuevo": new_m,
                    "b nuevo": new_b,
                    "MSE anterior": compute_mse(y, current_pred),
                    "MSE nuevo": compute_mse(y, new_pred),
                }
            ]
        )
        st.dataframe(iter_row, width="stretch", hide_index=True)
        g1, g2, g3, g4 = st.columns(4)
        run_steps = 0
        if g1.button("Siguiente iteración", key="guided_lr_btn_next_iter"):
            run_steps = 1
        if g2.button("Entrenar 10 iteraciones", key="guided_lr_btn_train_10"):
            run_steps = 10
        if g3.button("Entrenar 100 iteraciones", key="guided_lr_btn_train_100"):
            run_steps = 100
        if g4.button("Entrenar 1000 iteraciones", key="guided_lr_btn_train_1000"):
            run_steps = 1000
        if st.button("Reiniciar entrenamiento", key="guided_lr_btn_reset_simple"):
            st.session_state.guided_lr_gd_m = 0.03
            st.session_state.guided_lr_gd_b = 3.0
            st.session_state.guided_lr_gd_history = []
            st.rerun()
        if run_steps:
            m2, b2, hist, stable = train_simple_steps(
                x,
                y,
                st.session_state.guided_lr_gd_m,
                st.session_state.guided_lr_gd_b,
                lr,
                run_steps,
                st.session_state.guided_lr_gd_history,
            )
            if stable:
                st.session_state.guided_lr_gd_m = m2
                st.session_state.guided_lr_gd_b = b2
                st.session_state.guided_lr_gd_history = hist
                st.rerun()
            else:
                st.warning("El MSE se volvió NaN, infinito o demasiado grande. Bajá el learning rate.")
        gd_pred = predict_simple(x, st.session_state.guided_lr_gd_m, st.session_state.guided_lr_gd_b)
        gd_metrics = simple_metrics(y, gd_pred)
        metric_cards(
            [
                ("Iteración actual", f"{len(st.session_state.guided_lr_gd_history)}", None),
                ("m actual", f"{st.session_state.guided_lr_gd_m:.5f}", None),
                ("b actual", f"{st.session_state.guided_lr_gd_b:.4f}", None),
                ("MSE actual", f"{gd_metrics['mse']:.4f}", None),
                ("RMSE actual", f"{gd_metrics['rmse']:.4f}", None),
            ]
        )
        st.write("Así queda la regresión después del entrenamiento acumulado hasta este momento.")
        st.plotly_chart(
            plot_closed_line(
                df_small,
                st.session_state.guided_lr_gd_m,
                st.session_state.guided_lr_gd_b,
                True,
                "Línea aprendida con gradient descent y nuevos errores",
            ),
            width="stretch",
            key="guided_gd_trained_line",
        )
        st.write("Nuevas predicciones y errores usando los parámetros aprendidos por gradient descent.")
        st.dataframe(
            simple_manual_table(
                df_small,
                include_pred=True,
                include_error=True,
                m=st.session_state.guided_lr_gd_m,
                b=st.session_state.guided_lr_gd_b,
            ),
            width="stretch",
            hide_index=True,
        )
        h1, h2 = st.columns(2)
        h1.plotly_chart(history_line(st.session_state.guided_lr_gd_history, "mse", "MSE por iteración"), width="stretch", key="guided_mse_hist")
        h2.plotly_chart(history_line(st.session_state.guided_lr_gd_history, "rmse", "RMSE por iteración"), width="stretch", key="guided_rmse_hist")
        h3, h4 = st.columns(2)
        h3.plotly_chart(history_line(st.session_state.guided_lr_gd_history, "m", "m por iteración"), width="stretch", key="guided_m_hist")
        h4.plotly_chart(history_line(st.session_state.guided_lr_gd_history, "b", "b por iteración"), width="stretch", key="guided_b_hist")
        questions(["¿El error baja con las iteraciones?", "¿Siempre baja perfectamente?", "¿Qué pasa si el learning rate es muy alto?", "¿Qué pasa si es muy bajo?"])

    if st.session_state.guided_lr_step >= 10:
        section("Paso 10. Escalar a miles de datos")
        st.write("Con 10 filas podemos revisar los cálculos. Con 5000 sería imposible hacerlo a mano, pero la computadora repite las mismas operaciones.")
        n_rows = st.radio("Número de películas sintéticas", [10, 100, 1000, 5000], horizontal=True, key="guided_scale_rows")
        df_scale = scaled_movie_dataset(int(n_rows), 42)
        vals_scale = closed_form_values(
            df_scale.rename(columns={SIMPLE_FEATURE: SIMPLE_FEATURE, MANUAL_MOVIE_TARGET: MANUAL_MOVIE_TARGET}).assign(pelicula="")
        )
        scale_pred = predict_simple(df_scale[SIMPLE_FEATURE], vals_scale["m"], vals_scale["b"])
        scale_metrics = simple_metrics(df_scale[MANUAL_MOVIE_TARGET], scale_pred)
        metric_cards(
            [
                ("Cantidad de datos", f"{len(df_scale):,}", None),
                ("MAE", f"{scale_metrics['mae']:.3f}", None),
                ("RMSE", f"{scale_metrics['rmse']:.3f}", None),
            ]
        )
        st.plotly_chart(
            plot_closed_line(df_scale.assign(pelicula=""), vals_scale["m"], vals_scale["b"], len(df_scale) <= 100, f"Línea ajustada con {len(df_scale):,} películas"),
            width="stretch",
            key="guided_scale_plot",
        )
        questions(["¿La nube de puntos se ve más clara con más datos?", "¿El error cambia?", "¿La línea se vuelve más estable?", "¿Qué parte del proceso es igual con 10 y con 5000 datos?"])

    section("Ahora con 3 variables")
    df_multi = scaled_movie_dataset(1000, 42)
    st.write("Ahora ya no movemos una línea en 2D. El modelo aprende varios pesos al mismo tiempo.")
    st.latex(r"rating_{predicho}=b+w_1x_1+w_2x_2+w_3x_3")
    st.dataframe(
        df_small[["pelicula", SIMPLE_FEATURE, "popularidad", "presupuesto_millones", MANUAL_MOVIE_TARGET]].rename(
            columns={SIMPLE_FEATURE: "x1", "popularidad": "x2", "presupuesto_millones": "x3", MANUAL_MOVIE_TARGET: "y"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.info("Una predicción fila por fila se calcula como: b + w1*x1 + w2*x2 + w3*x3.")

    section("Normalización para las 3 variables")
    st.dataframe(
        pd.DataFrame(
            {
                "Variable": [SIMPLE_FEATURE, "popularidad", "presupuesto_millones"],
                "rango aproximado": ["0 a 100", "0 a 100", "1 a 300"],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    use_norm = st.checkbox("Normalizar variables", value=True, key="guided_multi_use_norm")
    X_norm, means, stds, norm_stats = normalize_features(df_multi, [SIMPLE_FEATURE, "popularidad", "presupuesto_millones"])
    st.dataframe(norm_stats if use_norm else df_multi[[SIMPLE_FEATURE, "popularidad", "presupuesto_millones"]].head(10), width="stretch", hide_index=True)
    X_multi = X_norm if use_norm else df_multi[[SIMPLE_FEATURE, "popularidad", "presupuesto_millones"]].astype(float)
    y_multi = df_multi[MANUAL_MOVIE_TARGET].to_numpy(dtype=float)

    section("Regresión múltiple con gradient descent")
    if "guided_multi_weights" not in st.session_state or len(st.session_state.guided_multi_weights) != 3:
        st.session_state.guided_multi_weights = np.zeros(3)
        st.session_state.guided_multi_bias = float(y_multi.mean())
        st.session_state.guided_multi_history = []
    multi_lr = st.slider("learning_rate múltiple", 0.000001, 0.1, 0.03, format="%.6f", key="guided_multi_lr")
    multi_pred = predict_multiple(X_multi, st.session_state.guided_multi_weights, st.session_state.guided_multi_bias)
    dw, db = compute_gradients_multiple(X_multi, y_multi, multi_pred)
    next_w, next_bias = update_multiple_params(st.session_state.guided_multi_weights, st.session_state.guided_multi_bias, dw, db, multi_lr)
    next_multi_pred = predict_multiple(X_multi, next_w, next_bias)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "w1": st.session_state.guided_multi_weights[0],
                    "w2": st.session_state.guided_multi_weights[1],
                    "w3": st.session_state.guided_multi_weights[2],
                    "bias": st.session_state.guided_multi_bias,
                    "gradiente w1": dw[0],
                    "gradiente w2": dw[1],
                    "gradiente w3": dw[2],
                    "gradiente bias": db,
                    "MSE actual": compute_mse(y_multi, multi_pred),
                    "MSE siguiente": compute_mse(y_multi, next_multi_pred),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    mb1, mb2, mb3, mb4 = st.columns(4)
    multi_steps = 0
    if mb1.button("Siguiente iteración múltiple", key="guided_multi_btn_next"):
        multi_steps = 1
    if mb2.button("Entrenar 100 iteraciones", key="guided_multi_btn_train_100"):
        multi_steps = 100
    if mb3.button("Entrenar 1000 iteraciones", key="guided_multi_btn_train_1000"):
        multi_steps = 1000
    if mb4.button("Reiniciar modelo múltiple", key="guided_multi_btn_reset"):
        st.session_state.guided_multi_weights = np.zeros(3)
        st.session_state.guided_multi_bias = float(y_multi.mean())
        st.session_state.guided_multi_history = []
        st.rerun()
    if multi_steps:
        weights, bias, hist, stable = train_multiple_steps(
            X_multi,
            y_multi,
            st.session_state.guided_multi_weights,
            st.session_state.guided_multi_bias,
            multi_lr,
            multi_steps,
            st.session_state.guided_multi_history,
        )
        if stable:
            st.session_state.guided_multi_weights = weights
            st.session_state.guided_multi_bias = bias
            st.session_state.guided_multi_history = hist
            st.rerun()
        else:
            st.warning("El MSE se volvió NaN, infinito o demasiado grande. Bajá el learning rate o activá normalización.")
    multi_pred = predict_multiple(X_multi, st.session_state.guided_multi_weights, st.session_state.guided_multi_bias)
    multi_mse = compute_mse(y_multi, multi_pred)
    metric_cards(
        [
            ("w1", f"{st.session_state.guided_multi_weights[0]:.4f}", SIMPLE_FEATURE),
            ("w2", f"{st.session_state.guided_multi_weights[1]:.4f}", "popularidad"),
            ("w3", f"{st.session_state.guided_multi_weights[2]:.4f}", "presupuesto"),
            ("bias", f"{st.session_state.guided_multi_bias:.4f}", None),
            ("RMSE", f"{np.sqrt(multi_mse):.4f}", None),
        ]
    )
    mh1, mh2 = st.columns(2)
    mh1.plotly_chart(history_line(st.session_state.guided_multi_history, "mse", "MSE múltiple"), width="stretch", key="guided_multi_mse")
    mh2.plotly_chart(history_line(st.session_state.guided_multi_history, "rmse", "RMSE múltiple"), width="stretch", key="guided_multi_rmse")
    wh1, wh2 = st.columns(2)
    wh1.plotly_chart(history_line(st.session_state.guided_multi_history, "w_1", "w1 por iteración"), width="stretch", key="guided_w1_hist")
    wh2.plotly_chart(history_line(st.session_state.guided_multi_history, "bias", "bias por iteración"), width="stretch", key="guided_bias_hist")
    questions(["¿Qué peso parece más importante?", "¿El presupuesto ayuda mucho o poco?", "¿El error baja más que con una sola variable?", "¿Qué pasa si no normalizamos?"])

    section("Comparación final con sklearn")
    compare_df = df_multi[[SIMPLE_FEATURE, "popularidad", "presupuesto_millones", MANUAL_MOVIE_TARGET]].rename(
        columns={"presupuesto_millones": "presupuesto_millones"}
    )
    compare_df = compare_df.assign(experiencia_director=0.0)
    manual_for_compare = predict_multiple(
        normalize_features(compare_df, MULTIPLE_FEATURES)[0],
        np.array([st.session_state.guided_multi_weights[0], st.session_state.guided_multi_weights[1], st.session_state.guided_multi_weights[2], 0.0]),
        st.session_state.guided_multi_bias,
    )
    st.dataframe(sklearn_comparison(compare_df, manual_for_compare), width="stretch", hide_index=True)
    st.write("scikit-learn hace esto de forma optimizada. Pero la idea base es la misma: usar datos, calcular errores y encontrar parámetros que reduzcan el error.")

    section("Mini quiz final")
    render_guided_quiz()

    section("Retos para la estudiante")
    retos = [
        ("Reto 1", "Calcula manualmente xy para una fila. ¿Coincide con la app?"),
        ("Reto 2", "Calcula manualmente x² para una fila. ¿Coincide con la app?"),
        ("Reto 3", "Mueve m y b para bajar RMSE. ¿Qué combinación te funcionó mejor?"),
        ("Reto 4", "Entrena 1000 iteraciones. ¿El error bajó?"),
        ("Reto 5", "Cambia de 10 datos a 5000. ¿Qué cambió?"),
        ("Reto 6", "Entrena con 3 variables. ¿Bajó el error comparado con una variable?"),
        ("Reto 7", "Apaga la normalización. ¿Qué pasa con el entrenamiento?"),
    ]
    cols = st.columns(2)
    for idx, (title, text) in enumerate(retos):
        with cols[idx % 2]:
            done = st.checkbox(title, key=f"guided_lr_reto_{idx}")
            st.caption(text if not done else f"Completado. {text}")


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

main_lab_tab, main_flow_tab, movie_regression_tab = st.tabs(
    ["Modelo de clasificación", "Flujo de datos del modelo", "Regresión lineal hecha a mano"]
)

with main_flow_tab:
    render_model_data_flow_section()

with movie_regression_tab:
    render_movie_regression_section()

with main_lab_tab:
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
