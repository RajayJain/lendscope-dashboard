"""Plotly figure builders for the LendScope dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Shared palette ---
RISK_GOOD = "#2e7d32"
RISK_MID = "#f9a825"
RISK_BAD = "#c62828"
ACCENT = "#1565c0"
PALETTE = px.colors.qualitative.Bold

LAYOUT = dict(
    margin=dict(l=10, r=10, t=50, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _risk_colors(values: pd.Series, invert: bool = False) -> list[str]:
    """Map a numeric series to a green→amber→red color scale."""
    v = values.astype(float).to_numpy()
    if v.max() == v.min():
        return [ACCENT] * len(v)
    norm = (v - v.min()) / (v.max() - v.min())
    if invert:
        norm = 1 - norm
    colors = []
    for n in norm:
        if n < 0.5:
            colors.append(_lerp(RISK_GOOD, RISK_MID, n * 2))
        else:
            colors.append(_lerp(RISK_MID, RISK_BAD, (n - 0.5) * 2))
    return colors


def _lerp(c1: str, c2: str, t: float) -> str:
    a = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(a[i] + (b[i]-a[i])*t):02x}" for i in range(3))


# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
def monthly_funded_vs_received(loans: pd.DataFrame) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    m = (
        loans.assign(month=loans["issue_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(funded=("funded_amount", "sum"), received=("total_payment", "sum"))
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["month"], y=m["funded"], name="Funded", mode="lines+markers",
        line=dict(color=ACCENT, width=3), fill="tozeroy",
        fillcolor="rgba(21,101,192,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=m["month"], y=m["received"], name="Received", mode="lines+markers",
        line=dict(color=RISK_GOOD, width=3),
    ))
    fig.update_layout(title="Monthly Funded vs. Received", **LAYOUT)
    fig.update_yaxes(tickprefix="$", tickformat=".2s")
    return fig


def loan_status_donut(loans: pd.DataFrame) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    counts = loans["loan_status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    color_map = {
        "Current": ACCENT, "Fully Paid": RISK_GOOD, "Charged Off": RISK_BAD,
    }
    fig = px.pie(
        counts, names="status", values="count", hole=0.55,
        color="status", color_discrete_map=color_map,
    )
    fig.update_traces(textinfo="percent+label", textposition="outside")
    fig.update_layout(title="Loan Status Mix", showlegend=False, **LAYOUT)
    return fig


# ----------------------------------------------------------------------------
# Portfolio quality
# ----------------------------------------------------------------------------
def chargeoff_by_grade(loans: pd.DataFrame) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    g = (
        loans.groupby("grade", as_index=False)
        .agg(applications=("loan_id", "count"),
             chargeoff=("is_bad", "mean"),
             avg_rate=("int_rate", "mean"))
        .sort_values("grade")
    )
    g["chargeoff_pct"] = g["chargeoff"] * 100
    fig = go.Figure(go.Bar(
        x=g["grade"], y=g["chargeoff_pct"],
        marker_color=_risk_colors(g["chargeoff_pct"]),
        text=[f"{v:.1f}%" for v in g["chargeoff_pct"]],
        textposition="outside",
        customdata=g[["applications", "avg_rate"]].round(2),
        hovertemplate=("Grade %{x}<br>Charge-off: %{y:.2f}%<br>"
                       "Applications: %{customdata[0]:,}<br>"
                       "Avg rate: %{customdata[1]:.2f}%<extra></extra>"),
    ))
    fig.update_layout(title="Charge-off Rate by Credit Grade", **LAYOUT)
    fig.update_yaxes(title="Charge-off %", ticksuffix="%")
    return fig


def rate_vs_chargeoff(loans: pd.DataFrame) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    g = (
        loans.groupby("grade", as_index=False)
        .agg(chargeoff=("is_bad", "mean"), avg_rate=("int_rate", "mean"))
        .sort_values("grade")
    )
    g["chargeoff_pct"] = g["chargeoff"] * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=g["grade"], y=g["chargeoff_pct"], name="Charge-off %",
        marker_color=RISK_BAD, opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=g["grade"], y=g["avg_rate"], name="Avg Interest Rate %",
        mode="lines+markers+text", text=[f"{v:.1f}%" for v in g["avg_rate"]],
        textposition="top center", line=dict(color=ACCENT, width=3),
    ), secondary_y=True)
    fig.update_layout(title="Is the Risk Premium Adequate? Rate vs. Charge-off", **LAYOUT)
    fig.update_yaxes(title_text="Charge-off %", ticksuffix="%", secondary_y=False)
    fig.update_yaxes(title_text="Avg Interest Rate %", ticksuffix="%", secondary_y=True)
    return fig


def chargeoff_by_purpose(loans: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    p = (
        loans.groupby("purpose", as_index=False)
        .agg(applications=("loan_id", "count"), chargeoff=("is_bad", "mean"))
    )
    p["chargeoff_pct"] = p["chargeoff"] * 100
    p = p.sort_values("chargeoff_pct", ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=p["chargeoff_pct"], y=p["purpose"], orientation="h",
        marker_color=_risk_colors(p["chargeoff_pct"]),
        text=[f"{v:.1f}%" for v in p["chargeoff_pct"]], textposition="outside",
        hovertemplate="%{y}<br>Charge-off: %{x:.2f}%<br>Apps: %{customdata:,}<extra></extra>",
        customdata=p["applications"],
    ))
    fig.update_layout(title="Charge-off Rate by Loan Purpose", **LAYOUT)
    fig.update_xaxes(title="Charge-off %", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


def dti_default_box(loans: pd.DataFrame) -> go.Figure:
    if loans.empty or "dti" not in loans.columns:
        return _empty("DTI data unavailable.")
    df = loans.dropna(subset=["dti"]).copy()
    df["Outcome"] = np.where(df["is_bad"], "Charged Off", "Good")
    fig = px.box(
        df, x="Outcome", y="dti", color="Outcome",
        color_discrete_map={"Charged Off": RISK_BAD, "Good": RISK_GOOD},
        points=False,
    )
    fig.update_layout(title="DTI Distribution by Loan Outcome", showlegend=False, **LAYOUT)
    fig.update_yaxes(title="Debt-to-Income (%)")
    return fig


# ----------------------------------------------------------------------------
# Geography
# ----------------------------------------------------------------------------
def top_states(loans: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    s = (
        loans.groupby("address_state", as_index=False)
        .agg(funded=("funded_amount", "sum"),
             applications=("loan_id", "count"),
             chargeoff=("is_bad", "mean"),
             avg_rate=("int_rate", "mean"))
        .sort_values("funded", ascending=False)
        .head(top_n)
    )
    s["chargeoff_pct"] = s["chargeoff"] * 100
    fig = go.Figure(go.Bar(
        x=s["funded"], y=s["address_state"], orientation="h",
        marker_color=_risk_colors(s["chargeoff_pct"]),
        text=[f"${v/1e6:,.1f}M" for v in s["funded"]], textposition="outside",
        customdata=s[["applications", "chargeoff_pct", "avg_rate"]].round(2),
        hovertemplate=("%{y}<br>Funded: $%{x:,.0f}<br>Applications: %{customdata[0]:,}"
                       "<br>Charge-off: %{customdata[1]:.2f}%"
                       "<br>Avg rate: %{customdata[2]:.2f}%<extra></extra>"),
    ))
    fig.update_layout(
        title="Top States by Funded Amount (color = charge-off risk)", **LAYOUT
    )
    fig.update_xaxes(tickprefix="$", tickformat=".2s")
    fig.update_yaxes(autorange="reversed")
    return fig


def state_choropleth(loans: pd.DataFrame) -> go.Figure:
    if loans.empty:
        return _empty("No loans match the current filters.")
    s = (
        loans.groupby("address_state", as_index=False)
        .agg(chargeoff=("is_bad", "mean"), funded=("funded_amount", "sum"))
    )
    s["chargeoff_pct"] = s["chargeoff"] * 100
    fig = px.choropleth(
        s, locations="address_state", locationmode="USA-states",
        color="chargeoff_pct", scope="usa",
        color_continuous_scale=["#2e7d32", "#f9a825", "#c62828"],
        labels={"chargeoff_pct": "Charge-off %"},
        hover_data={"funded": ":,.0f"},
    )
    fig.update_layout(title="Charge-off Rate by State", **LAYOUT)
    return fig


# ----------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------
def customer_360(loans: pd.DataFrame, txns: pd.DataFrame, customers: pd.DataFrame,
                 top_n: int = 25) -> pd.DataFrame:
    loan_agg = loans.groupby("customer_id", as_index=False).agg(
        loan_funded=("funded_amount", "sum"),
        loan_received=("total_payment", "sum"),
        n_loans=("loan_id", "count"),
        n_chargeoff=("is_bad", "sum"),
    )
    txn_agg = txns.groupby("customer_id", as_index=False).agg(
        card_spend=("amount", "sum"),
        n_txns=("transaction_id", "count"),
        n_fraud=("is_fraud", "sum"),
    )
    base = customers[["customer_id", "first_name", "last_name",
                      "address_state", "credit_score"]]
    out = (
        base.merge(loan_agg, on="customer_id", how="left")
            .merge(txn_agg, on="customer_id", how="left")
            .fillna({"loan_funded": 0, "loan_received": 0, "card_spend": 0,
                     "n_loans": 0, "n_chargeoff": 0, "n_txns": 0, "n_fraud": 0})
    )
    out["relationship_value"] = out["loan_funded"] + out["card_spend"]
    return out.sort_values("relationship_value", ascending=False).head(top_n)


def relationship_scatter(loans: pd.DataFrame, txns: pd.DataFrame) -> go.Figure:
    loan_agg = loans.groupby("customer_id", as_index=False).agg(
        loan_funded=("funded_amount", "sum"), n_chargeoff=("is_bad", "sum"))
    txn_agg = txns.groupby("customer_id", as_index=False).agg(
        card_spend=("amount", "sum"))
    m = loan_agg.merge(txn_agg, on="customer_id", how="outer").fillna(0)
    if m.empty:
        return _empty("No customer activity matches the filters.")
    m["segment"] = np.where(m["n_chargeoff"] > 0, "Has charge-off", "Clean")
    fig = px.scatter(
        m, x="loan_funded", y="card_spend", color="segment", size="card_spend",
        color_discrete_map={"Clean": ACCENT, "Has charge-off": RISK_BAD},
        opacity=0.55, log_x=True, log_y=True,
    )
    fig.update_layout(title="Loan Value vs. Card Value per Customer", **LAYOUT)
    fig.update_xaxes(title="Total Loan Funded ($, log)", tickprefix="$", tickformat=".2s")
    fig.update_yaxes(title="Total Card Spend ($, log)", tickprefix="$", tickformat=".2s")
    return fig


def score_distribution(customers: pd.DataFrame) -> go.Figure:
    if customers.empty:
        return _empty("No customers match the current filters.")
    fig = px.histogram(
        customers, x="credit_score", nbins=40,
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(title="Credit Score Distribution", **LAYOUT, bargap=0.05)
    fig.update_xaxes(title="Credit Score")
    fig.update_yaxes(title="Customers")
    return fig


# ----------------------------------------------------------------------------
# Cards & fraud
# ----------------------------------------------------------------------------
def fraud_by_channel_card(txns: pd.DataFrame) -> go.Figure:
    if txns.empty:
        return _empty("No transactions match the current filters.")
    g = (
        txns.groupby(["card_type", "channel"], as_index=False)
        .agg(txns=("transaction_id", "count"),
             fraud=("is_fraud", "sum"),
             fraud_amt=("fraud_amount", "sum"))
    )
    g["fraud_rate"] = np.where(g["txns"] > 0, g["fraud"] / g["txns"] * 100, 0)
    fig = px.bar(
        g, x="channel", y="fraud_rate", color="card_type",
        barmode="group", text=[f"{v:.2f}%" for v in g["fraud_rate"]],
        color_discrete_sequence=[ACCENT, "#7b1fa2"],
        custom_data=["txns", "fraud", "fraud_amt"],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=("%{x} · %{fullData.name}<br>Fraud rate: %{y:.3f}%"
                       "<br>Transactions: %{customdata[0]:,}"
                       "<br>Fraud count: %{customdata[1]:,}"
                       "<br>Fraud exposure: $%{customdata[2]:,.0f}<extra></extra>"),
    )
    fig.update_layout(title="Fraud Rate by Channel × Card Type", **LAYOUT)
    fig.update_yaxes(title="Fraud Rate %", ticksuffix="%")
    return fig


def monthly_fraud_trend(txns: pd.DataFrame) -> go.Figure:
    if txns.empty:
        return _empty("No transactions match the current filters.")
    m = (
        txns.assign(month=txns["transaction_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(txns=("transaction_id", "count"),
             fraud=("is_fraud", "sum"),
             fraud_amt=("fraud_amount", "sum"))
    )
    m["fraud_rate"] = np.where(m["txns"] > 0, m["fraud"] / m["txns"] * 100, 0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=m["month"], y=m["fraud_amt"], name="Fraud exposure ($)",
        marker_color="rgba(198,40,40,0.35)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=m["month"], y=m["fraud_rate"], name="Fraud rate %",
        mode="lines+markers", line=dict(color=RISK_BAD, width=3),
    ), secondary_y=True)
    fig.update_layout(title="Monthly Fraud Exposure & Rate", **LAYOUT)
    fig.update_yaxes(title_text="Fraud exposure ($)", tickprefix="$",
                     tickformat=".2s", secondary_y=False)
    fig.update_yaxes(title_text="Fraud rate %", ticksuffix="%", secondary_y=True)
    return fig


def merchant_fraud(txns: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if txns.empty or "merchant_category" not in txns.columns:
        return _empty("Merchant category data unavailable.")
    g = (
        txns.groupby("merchant_category", as_index=False)
        .agg(txns=("transaction_id", "count"),
             fraud=("is_fraud", "sum"),
             fraud_amt=("fraud_amount", "sum"))
    )
    g["fraud_rate"] = np.where(g["txns"] > 0, g["fraud"] / g["txns"] * 100, 0)
    g = g.sort_values("fraud_rate", ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=g["fraud_rate"], y=g["merchant_category"], orientation="h",
        marker_color=_risk_colors(g["fraud_rate"]),
        text=[f"{v:.2f}%" for v in g["fraud_rate"]], textposition="outside",
        customdata=g[["txns", "fraud_amt"]],
        hovertemplate=("%{y}<br>Fraud rate: %{x:.3f}%<br>"
                       "Transactions: %{customdata[0]:,}<br>"
                       "Fraud exposure: $%{customdata[1]:,.0f}<extra></extra>"),
    ))
    fig.update_layout(title="Fraud Rate by Merchant Category", **LAYOUT)
    fig.update_xaxes(title="Fraud rate %", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


# ----------------------------------------------------------------------------
# Cross-product
# ----------------------------------------------------------------------------
def score_band_dual_risk(loans: pd.DataFrame, txns: pd.DataFrame) -> go.Figure:
    band_order = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]

    l = loans.dropna(subset=["score_band"])
    l = l.groupby("score_band", observed=True).agg(
        apps=("loan_id", "count"), chargeoff=("is_bad", "mean")).reset_index()
    l["chargeoff_pct"] = l["chargeoff"] * 100

    t = txns.dropna(subset=["score_band"])
    t = t.groupby("score_band", observed=True).agg(
        txns=("transaction_id", "count"), fraud=("is_fraud", "mean")).reset_index()
    t["fraud_pct"] = t["fraud"] * 100

    m = pd.DataFrame({"score_band": band_order}).merge(l, how="left").merge(t, how="left")
    m = m.fillna(0)
    # keep only bands present in the data
    present = [b for b in band_order if b in set(loans["score_band"].dropna().unique())
               | set(txns["score_band"].dropna().unique())]
    m = m[m["score_band"].isin(present)]

    if m.empty:
        return _empty("No score-band data matches the filters.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=m["score_band"], y=m["chargeoff_pct"], name="Loan charge-off %",
        marker_color=RISK_BAD, opacity=0.85,
        text=[f"{v:.2f}%" for v in m["chargeoff_pct"]], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=m["score_band"], y=m["fraud_pct"], name="Card fraud %",
        mode="lines+markers", line=dict(color=ACCENT, width=3),
        marker=dict(size=10),
    ), secondary_y=True)
    fig.update_layout(title="Cross-Product Risk by Credit Score Band", **LAYOUT)
    fig.update_yaxes(title_text="Loan charge-off %", ticksuffix="%", secondary_y=False)
    fig.update_yaxes(title_text="Card fraud %", ticksuffix="%", secondary_y=True)
    return fig


def chargeoff_vs_fraud_quadrant(loans: pd.DataFrame, txns: pd.DataFrame) -> go.Figure:
    """Do charge-off customers also commit card fraud?"""
    loan_risk = (
        loans.groupby("customer_id", as_index=False)
        .agg(has_chargeoff=("is_bad", "max"))
    )
    t = txns.merge(loan_risk, on="customer_id", how="left")
    t["has_chargeoff"] = t["has_chargeoff"].fillna(False)
    g = t.groupby("has_chargeoff", as_index=False).agg(
        txns=("transaction_id", "count"), fraud=("is_fraud", "mean"))
    g["fraud_pct"] = g["fraud"] * 100
    g["label"] = np.where(g["has_chargeoff"], "Has charged-off loan", "No charge-off")

    fig = go.Figure(go.Bar(
        x=g["label"], y=g["fraud_pct"],
        marker_color=[ACCENT, RISK_BAD][:len(g)],
        text=[f"{v:.3f}%" for v in g["fraud_pct"]], textposition="outside",
        customdata=g["txns"],
        hovertemplate="%{x}<br>Fraud rate: %{y:.3f}%<br>Transactions: %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(title="Card Fraud Rate by Loan-Default History", **LAYOUT)
    fig.update_yaxes(title="Card fraud rate %", ticksuffix="%")
    return fig


# ----------------------------------------------------------------------------
def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=14, color="#888"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**{**LAYOUT, "title": ""}, height=320)
    return fig