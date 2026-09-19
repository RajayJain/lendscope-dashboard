"""Plotly figures for the XGBoost scoring tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACCENT = "#1565c0"
RISK_GOOD = "#2e7d32"
RISK_MID = "#f9a825"
RISK_BAD = "#c62828"

LAYOUT = dict(
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def roc_curve_plot(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC = {auc:.3f})",
        line=dict(color=ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.10)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#888", dash="dash"),
    ))
    fig.update_layout(title="ROC Curve", **LAYOUT)
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return fig


def pr_curve_plot(prec, rec, ap: float, baseline: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name=f"XGBoost (PR-AUC = {ap:.3f})",
        line=dict(color=RISK_BAD, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline], mode="lines",
        name=f"Baseline ({baseline:.3f})",
        line=dict(color="#888", dash="dash"),
    ))
    fig.update_layout(title="Precision–Recall Curve", **LAYOUT)
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return fig


def confusion_matrix_plot(cm: np.ndarray, threshold: float) -> go.Figure:
    labels = ["Good (0)", "Bad (1)"]
    z = cm
    text = [[f"{v:,}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
        colorscale=[[0, "#ffffff"], [1, "#1565c0"]], showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix @ threshold = {threshold:.2f}",
        **LAYOUT,
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return fig


def feature_importance_plot(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    d = importance.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["gain_share"], y=d["feature"], orientation="h",
        marker_color=ACCENT,
        text=[f"{v*100:.1f}%" for v in d["gain_share"]],
        textposition="outside",
        hovertemplate="%{y}<br>Gain share: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=f"Top {top_n} Features by Gain", **LAYOUT)
    fig.update_xaxes(title="Share of total gain", tickformat=".0%")
    return fig


def shap_waterfall(explanation: dict, top_k: int = 12) -> go.Figure:
    """Horizontal bar chart of the top-K log-odds contributions."""
    names = explanation["features"]
    values = explanation["contributions"]
    base = explanation["base"]
    prob = explanation["prob"]

    order = np.argsort(np.abs(values))[::-1][:top_k]
    pairs = sorted(zip([names[i] for i in order], values[order]),
                   key=lambda p: p[1])
    feat = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [RISK_BAD if v > 0 else RISK_GOOD for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feat, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        hovertemplate="%{y}<br>Contribution: %{x:+.4f} log-odds<extra></extra>",
    ))
    fig.update_layout(
        title=(f"Why this score?  ·  base={base:+.3f}  →  "
               f"final log-odds={base + sum(vals):+.3f}  ·  P(default)={prob:.2%}"),
        showlegend=False,
        margin=dict(l=10, r=40, t=70, b=10),
    )
    fig.update_xaxes(title="Log-odds contribution  (red ↑ risk, green ↓ risk)")
    fig.update_yaxes(autorange="reversed")
    return fig


def score_gauge(prob: float, approve_below: float, decline_above: float) -> go.Figure:
    color = (RISK_GOOD if prob < approve_below
             else RISK_MID if prob < decline_above
             else RISK_BAD)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, approve_below * 100], "color": "rgba(46,125,50,0.18)"},
                {"range": [approve_below * 100, decline_above * 100],
                 "color": "rgba(249,168,37,0.20)"},
                {"range": [decline_above * 100, 100], "color": "rgba(198,40,40,0.18)"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        title={"text": "Probability of Default", "x": 0.5},
    )
    return fig