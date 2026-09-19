"""LendScope — unified loan & card portfolio analytics dashboard."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
import model as ml
import model_charts as mc
from model import APPROVE_BELOW, DECLINE_ABOVE, DEFAULTS, FEATURES, NUMERIC_FEATURES
from model import CATEGORICAL_FEATURES, decision_band
from data_loader import (
    apply_filters,
    filter_options,
    load_customers,
    load_loans,
    load_transactions,
    money,
)

st.set_page_config(
    page_title="LendScope | Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
customers = load_customers()
loans_all = load_loans()
txns_all = load_transactions()
opts = filter_options(loans_all, txns_all)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 LendScope")
    st.caption("Loan & card portfolio analytics")
    st.divider()

    st.subheader("Filters")

    all_states = sorted(customers["address_state"].dropna().unique())
    states = st.multiselect("State", all_states, default=[], placeholder="All states")

    score_range = st.slider(
        "Credit score", 300, 850, (300, 850), step=10,
    )

    ownership = st.multiselect(
        "Home ownership", opts["ownership"], default=[],
        placeholder="All types",
    )

    grades = st.multiselect(
        "Loan grade", opts["grades"], default=[], placeholder="All grades",
    )

    loan_statuses = st.multiselect(
        "Loan status", opts["loan_statuses"], default=[], placeholder="All statuses",
    )

    purposes = st.multiselect(
        "Loan purpose", opts["purposes"], default=[], placeholder="All purposes",
    )

    card_types = st.multiselect(
        "Card type", opts["card_types"], default=[], placeholder="All card types",
    )

    channels = st.multiselect(
        "Channel", opts["channels"], default=[], placeholder="All channels",
    )

    date_range = st.date_input(
        "Date range (issue / transaction)",
        value=(pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-12-31").date()),
        min_value=pd.Timestamp("2024-01-01").date(),
        max_value=pd.Timestamp("2027-12-31").date(),
    )

    st.divider()
    if st.button("Reset filters", width='stretch'):
        st.rerun()

    st.caption("Data source: CSV (`data/`) or Postgres via `LENDSCOPE_DB_URL`.")

filters = {
    "states": states,
    "score_range": score_range if score_range != (300, 850) else None,
    "ownership": ownership,
    "grades": grades,
    "loan_statuses": loan_statuses,
    "purposes": purposes,
    "card_types": card_types,
    "channels": channels,
    "date_range": date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None,
}

customers_f, loans_f, txns_f = apply_filters(customers, loans_all, txns_all, filters)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("LendScope — Loan & Card Portfolio Analytics")
st.caption(
    f"{len(customers_f):,} customers · {len(loans_f):,} loans · "
    f"{len(txns_f):,} card transactions"
)

if loans_f.empty and txns_f.empty:
    st.warning("No data matches the current filters. Try widening your selection.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
total_funded = loans_f["funded_amount"].sum()
total_received = loans_f["total_payment"].sum()
rf_ratio = (total_received / total_funded * 100) if total_funded else 0
chargeoff_rate = loans_f["is_bad"].mean() * 100 if not loans_f.empty else 0
avg_rate = loans_f["int_rate"].mean() if not loans_f.empty else 0
fraud_rate = txns_f["is_fraud"].mean() * 100 if not txns_f.empty else 0
fraud_exposure = txns_f["fraud_amount"].sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Funded", money(total_funded))
k2.metric("Total Received", money(total_received), f"{rf_ratio:.1f}% of funded")
k3.metric("Applications", f"{len(loans_f):,}")
k4.metric("Charge-off Rate", f"{chargeoff_rate:.2f}%")
k5.metric("Avg Interest Rate", f"{avg_rate:.2f}%")
k6.metric("Card Fraud Rate", f"{fraud_rate:.3f}%", money(fraud_exposure) + " exposure")

st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
(tab_overview, tab_quality, tab_geo, tab_customers,
 tab_cards, tab_cross, tab_score, tab_data) = st.tabs(
    ["📈 Overview", "⚠️ Portfolio Quality", "🗺️ Geography",
     "👥 Customers", "💳 Cards & Fraud", "🔀 Cross-Product",
     "🎯 Default Scorer", "🧾 Data"]
)
# --- Overview ---------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(ch.monthly_funded_vs_received(loans_f),
                        width='stretch')
    with c2:
        st.plotly_chart(ch.loan_status_donut(loans_f), width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(ch.score_distribution(customers_f), width='stretch')
    with c4:
        st.plotly_chart(ch.dti_default_box(loans_f), width='stretch')

# --- Portfolio quality ------------------------------------------------------
with tab_quality:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(ch.chargeoff_by_grade(loans_f), width='stretch')
    with c2:
        st.plotly_chart(ch.rate_vs_chargeoff(loans_f), width='stretch')

    st.plotly_chart(ch.chargeoff_by_purpose(loans_f), width='stretch')

    with st.expander("Grade-level detail table"):
        tbl = (
            loans_f.groupby("grade", as_index=False)
            .agg(applications=("loan_id", "count"),
                 funded=("funded_amount", "sum"),
                 received=("total_payment", "sum"),
                 avg_rate=("int_rate", "mean"),
                 chargeoff_rate=("is_bad", "mean"))
            .sort_values("grade")
        )
        tbl["funded"] = tbl["funded"].map(money)
        tbl["received"] = tbl["received"].map(money)
        tbl["avg_rate"] = tbl["avg_rate"].round(2).astype(str) + "%"
        tbl["chargeoff_rate"] = (tbl["chargeoff_rate"] * 100).round(2).astype(str) + "%"
        st.dataframe(tbl, width='stretch', hide_index=True)

# --- Geography --------------------------------------------------------------
with tab_geo:
    st.plotly_chart(ch.top_states(loans_f), width='stretch')
    st.plotly_chart(ch.state_choropleth(loans_f), width='stretch')

# --- Customers --------------------------------------------------------------
with tab_customers:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(ch.relationship_scatter(loans_f, txns_f),
                        width='stretch')
    with c2:
        top_n = st.slider("Top N customers by relationship value", 10, 100, 25, 5)
        c360 = ch.customer_360(loans_f, txns_f, customers_f, top_n=top_n)
        st.markdown(f"**Top {top_n} customers by relationship value**")
        st.dataframe(
            c360.assign(
                relationship_value=lambda d: d["relationship_value"].map(money),
                loan_funded=lambda d: d["loan_funded"].map(money),
                loan_received=lambda d: d["loan_received"].map(money),
                card_spend=lambda d: d["card_spend"].map(money),
            ),
            width='stretch',
            hide_index=True,
            height=520,
        )

# --- Cards & Fraud ----------------------------------------------------------
with tab_cards:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(ch.fraud_by_channel_card(txns_f), width='stretch')
    with c2:
        st.plotly_chart(ch.merchant_fraud(txns_f), width='stretch')

    st.plotly_chart(ch.monthly_fraud_trend(txns_f), width='stretch')

    with st.expander("Channel × card type detail"):
        tbl = (
            txns_f.groupby(["card_type", "channel"], as_index=False)
            .agg(transactions=("transaction_id", "count"),
                 fraud_count=("is_fraud", "sum"),
                 fraud_amount=("fraud_amount", "sum"))
        )
        tbl["fraud_rate"] = (
            tbl["fraud_count"] / tbl["transactions"] * 100
        ).round(3).astype(str) + "%"
        tbl["fraud_amount"] = tbl["fraud_amount"].map(money)
        st.dataframe(tbl, width='stretch', hide_index=True)

# --- Cross-product ----------------------------------------------------------
with tab_cross:
    st.plotly_chart(ch.score_band_dual_risk(loans_f, txns_f),
                    width='stretch')
    st.plotly_chart(ch.chargeoff_vs_fraud_quadrant(loans_f, txns_f),
                    width='stretch')

    st.info(
        "**Reading this chart:** if the two bars are nearly equal, loan default "
        "status is *not* a good proxy for card fraud — meaning a unified risk "
        "score should combine both signals rather than derive one from the other."
    )

# --- Default Scorer ---------------------------------------------------------
with tab_score:
    st.markdown("### Default Prediction — XGBoost Classifier")
    st.caption(
        "Trained on the full loans book (all filters ignored) so scores remain "
        "stable. Model retrains automatically when the underlying data changes."
    )

    # ---- Train / cache ----------------------------------------------------
    cache_key = f"{len(loans_all)}-{int(loans_all['is_bad'].sum())}-v1"

    @st.cache_resource(show_spinner="Training XGBoost classifier…")
    def _get_model(_key: str, _loans: pd.DataFrame):
        return ml.train_model(_loans)

    model, meta, eval_bundle = _get_model(cache_key, loans_all)

    with st.expander("Model card", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC AUC", f"{meta['auc']:.3f}")
        c2.metric("PR AUC (avg precision)", f"{meta['average_precision']:.3f}")
        c3.metric("F1-optimal threshold", f"{meta['best_threshold']:.3f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Train rows", f"{meta['n_train']:,}")
        c5.metric("Test rows", f"{meta['n_test']:,}")
        c6.metric("Class balance (train)", f"{meta['positive_rate']*100:.2f}% bad")

        st.caption(
            f"Trained at {meta['trained_at']} UTC · "
            f"scale_pos_weight = {meta['scale_pos_weight']:.2f} · "
            f"{len(FEATURES)} features "
            f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)"
        )

        col_a, col_b = st.columns([1, 1])
        if col_a.button("💾 Save model to disk", width='stretch'):
            ml.save_model(model, meta)
            col_a.success(f"Saved to `{ml.MODEL_PATH}`")
        if col_b.button("♻️ Force retrain", width='stretch'):
            st.cache_resource.clear()
            st.rerun()

    st.divider()

    # ---- Performance ------------------------------------------------------
    st.markdown("#### Model Performance (held-out 20%)")
    p1, p2 = st.columns(2)
    with p1:
        fpr, tpr = eval_bundle["roc"]
        st.plotly_chart(mc.roc_curve_plot(fpr, tpr, eval_bundle["auc"]),
                        width='stretch')
    with p2:
        prec, rec = eval_bundle["pr"]
        st.plotly_chart(
            mc.pr_curve_plot(prec, rec, eval_bundle["average_precision"],
                             meta["positive_rate"]),
            width='stretch',
        )

    p3, p4 = st.columns([1, 1])
    with p3:
        st.plotly_chart(
            mc.confusion_matrix_plot(eval_bundle["confusion"],
                                     eval_bundle["threshold"]),
            width='stretch',
        )
    with p4:
        st.plotly_chart(
            mc.feature_importance_plot(ml.global_feature_importance(model),
                                       top_n=15),
            width='stretch',
        )

    st.divider()

    # ---- Interactive scorer ----------------------------------------------
    st.markdown("#### Score a New Application")
    st.caption(
        "Change any input and press **Score application**. "
        "The gauge and SHAP chart update together."
    )

    with st.form("scorer_form", border=True):
        f1, f2, f3 = st.columns(3)

        with f1:
            credit_score = st.slider("Credit score", 300, 850,
                                     int(DEFAULTS["credit_score"]), 1)
            grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"],
                                 index=2)
            int_rate = st.number_input("Interest rate (%)", 1.0, 40.0,
                                       float(DEFAULTS["int_rate"]), 0.1)
            term_months = st.selectbox("Term (months)", [36, 60], index=0)
            funded_amount = st.number_input("Funded amount ($)", 500.0, 40000.0,
                                            float(DEFAULTS["funded_amount"]), 100.0)
            installment = st.number_input("Monthly installment ($)", 10.0, 2000.0,
                                          float(DEFAULTS["installment"]), 5.0)

        with f2:
            annual_income = st.number_input("Annual income ($)", 10000.0, 500000.0,
                                            float(DEFAULTS["annual_income"]), 1000.0)
            dti = st.slider("DTI (%)", 0.0, 60.0, float(DEFAULTS["dti"]), 0.5)
            employment_length = st.slider("Employment length (yrs)", 0.0, 30.0,
                                          float(DEFAULTS["employment_length"]), 0.5)
            delinq_2yrs = st.slider("Delinquencies (2yrs)", 0, 10,
                                    int(DEFAULTS["delinq_2yrs"]))
            open_accounts = st.slider("Open accounts", 0, 50,
                                      int(DEFAULTS["open_accounts"]))
            revol_util = st.slider("Revolving utilization (%)", 0.0, 120.0,
                                   float(DEFAULTS["revol_util"]), 1.0)

        with f3:
            purpose = st.selectbox(
                "Purpose",
                ["debt_consolidation", "credit_card", "home_improvement",
                 "major_purchase", "medical", "small_business", "car",
                 "vacation", "moving", "other"],
                index=0,
            )
            home_ownership = st.selectbox("Home ownership",
                                          ["RENT", "MORTGAGE", "OWN", "OTHER"],
                                          index=0)
            address_state = st.selectbox(
                "State",
                sorted(customers["address_state"].dropna().unique()),
                index=0,
            )
            verification_status = st.selectbox(
                "Verification", ["Verified", "Source Verified", "Not Verified"],
                index=0,
            )
            application_type = st.selectbox("Application type",
                                            ["Individual", "Joint"], index=0)

        submitted = st.form_submit_button("🎯 Score application",
                                          width='stretch', type="primary")

    # Persist last submission across widget reruns (e.g., slider tweaks)
    if submitted or "last_application" not in st.session_state:
        st.session_state["last_application"] = {
            "credit_score": credit_score,
            "grade": grade,
            "int_rate": int_rate,
            "term_months": term_months,
            "funded_amount": funded_amount,
            "installment": installment,
            "annual_income": annual_income,
            "dti": dti,
            "employment_length": employment_length,
            "delinq_2yrs": delinq_2yrs,
            "open_accounts": open_accounts,
            "revol_util": revol_util,
            "purpose": purpose,
            "home_ownership": home_ownership,
            "address_state": address_state,
            "verification_status": verification_status,
            "application_type": application_type,
        }

    application = st.session_state["last_application"]
    explanation = ml.score_application(model, meta, application)
    prob = explanation["prob"]
    band_label, band_color = decision_band(prob)

    st.divider()
    st.markdown("#### Result")

    r1, r2 = st.columns([1, 1.4])
    with r1:
        st.plotly_chart(
            mc.score_gauge(prob, APPROVE_BELOW, DECLINE_ABOVE),
            width='stretch',
        )
        st.markdown(
            f"<div style='text-align:center;padding:8px 12px;border-radius:8px;"
            f"background:{band_color};color:white;font-weight:600;"
            f"font-size:1.05rem;letter-spacing:0.3px'>{band_label}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Bands — Approve < {APPROVE_BELOW:.0%} · "
            f"Review {APPROVE_BELOW:.0%}–{DECLINE_ABOVE:.0%} · "
            f"Decline ≥ {DECLINE_ABOVE:.0%}"
        )

    with r2:
        st.plotly_chart(mc.shap_waterfall(explanation, top_k=12),
                        width='stretch')

    with st.expander("Full feature contributions & inputs"):
        contrib_df = pd.DataFrame({
            "feature": explanation["features"],
            "value": [application[f] for f in explanation["features"]],
            "contribution_logodds": explanation["contributions"],
        }).sort_values("contribution_logodds", key=lambda s: s.abs(), ascending=False)
        contrib_df["direction"] = np.where(
            contrib_df["contribution_logodds"] > 0, "↑ risk", "↓ risk"
        )
        st.dataframe(contrib_df, width='stretch', hide_index=True)
        st.caption(
            f"Base (bias) = {explanation['base']:+.4f} · "
            f"Final log-odds = {explanation['log_odds']:+.4f} · "
            f"P(default) = {prob:.4f}"
        )

# --- Data -------------------------------------------------------------------
with tab_data:
    st.markdown("### Filtered data preview")
    st.caption("Use these to sanity-check the numbers behind the charts.")

    t1, t2, t3 = st.tabs(["Customers", "Loans", "Card transactions"])
    with t1:
        st.dataframe(customers_f.head(500), width='stretch', height=420)
    with t2:
        st.dataframe(loans_f.head(500), width='stretch', height=420)
    with t3:
        st.dataframe(txns_f.head(500), width='stretch', height=420)

    st.download_button(
        "⬇️ Download filtered loans as CSV",
        loans_f.to_csv(index=False).encode("utf-8"),
        file_name="lendscope_loans_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "LendScope · FY2026 · Synthetic/anonymized data for demonstration. "
    "Not financial advice."
)