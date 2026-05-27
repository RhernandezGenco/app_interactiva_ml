from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_OK = "#2a9d8f"
COLOR_WARN = "#e76f51"
COLOR_BLUE = "#457b9d"


def target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["devuelto"].value_counts().rename(index={0: "No devuelto", 1: "Devuelto"}).reset_index()
    counts.columns = ["clase", "ventas"]
    return px.bar(counts, x="clase", y="ventas", color="clase", color_discrete_sequence=[COLOR_OK, COLOR_WARN])


def missing_values(df: pd.DataFrame) -> go.Figure:
    missing = df.isna().sum().reset_index()
    missing.columns = ["columna", "nulos"]
    missing = missing.sort_values("nulos", ascending=False)
    return px.bar(missing, x="columna", y="nulos", color_discrete_sequence=[COLOR_BLUE])


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["devuelto"], errors="ignore")
    corr = numeric.corr()
    return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")


def metrics_comparison(df: pd.DataFrame) -> go.Figure:
    long_df = df.melt(id_vars="experimento", var_name="metrica", value_name="valor")
    return px.bar(long_df, x="metrica", y="valor", color="experimento", barmode="group", range_y=[0, 1])


def split_bar(train_pct: float, test_pct: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Datos"], x=[train_pct], name=f"Entrenamiento {train_pct:.0f}%", orientation="h", marker_color=COLOR_OK))
    fig.add_trace(go.Bar(y=["Datos"], x=[test_pct], name=f"Prueba {test_pct:.0f}%", orientation="h", marker_color=COLOR_WARN))
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], title="%"), height=170, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def overfitting_line(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["max_depth"], y=df["accuracy_train"], mode="lines+markers", name="Train", line=dict(color=COLOR_OK)))
    fig.add_trace(go.Scatter(x=df["max_depth"], y=df["accuracy_test"], mode="lines+markers", name="Test", line=dict(color=COLOR_WARN)))
    fig.update_layout(xaxis_title="Profundidad del árbol", yaxis_title="Accuracy", yaxis=dict(range=[0, 1]))
    return fig


def confusion_matrix_plot(matrix: np.ndarray) -> go.Figure:
    labels = [["VN", "FP"], ["FN", "VP"]]
    text = [[f"{labels[i][j]}<br>{matrix[i, j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(data=go.Heatmap(z=matrix, x=["Predice 0", "Predice 1"], y=["Real 0", "Real 1"], text=text, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def roc_curve_plot(roc_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc_df["fpr"], y=roc_df["tpr"], mode="lines", name="Modelo", line=dict(color=COLOR_BLUE)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color="#999", dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1]))
    return fig


def feature_importance_plot(df: pd.DataFrame) -> go.Figure:
    ordered = df.sort_values("importancia", ascending=True)
    return px.bar(ordered, x="importancia", y="variable", orientation="h", color_discrete_sequence=[COLOR_BLUE])


def before_after_scaling(stats_before: pd.DataFrame, stats_after: pd.DataFrame) -> go.Figure:
    before = stats_before.assign(estado="Antes")
    after = stats_after.assign(estado="Después")
    combined = pd.concat([before, after], ignore_index=True)
    return px.bar(combined, x="variable", y="std", color="estado", barmode="group", title="Desviación estándar antes y después")
