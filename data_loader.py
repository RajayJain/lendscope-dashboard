"""Data loading, caching, and filtering for the LendScope dashboard."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

# ----------------------------------------------------------------------------
# Schema assumptions — adjust these if your real columns differ.
# ----------------------------------------------------------------------------
LOAN_DATE_COL = "issue_date"
TXN_DATE_COL = "transaction_date"
BAD_STATUS = "Charged Off"
GOOD_STATUSES = ("Fully Paid", "Current")

SCORE_BANDS = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]
SCORE_BINS = [0, 580, 670, 740, 800, 851]


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading customers…")
def load_customers() -> pd.DataFrame:
    return _read("customers.csv", parse_dates=["signup_date"])


@st.cache_data(show_spinner="Loading loans…")
def load_loans() -> pd.DataFrame:
    df = _read("loans.csv", parse_dates=[LOAN_DATE_COL])
    df["is_bad"] = df["loan_status"].eq(BAD_STATUS)
    df["is_good"] = df["loan_status"].isin(GOOD_STATUSES)
    df["score_band"] = pd.cut(
        df["credit_score"], bins=SCORE_BINS, labels=SCORE_BANDS, right=False
    )
    return df


@st.cache_data(show_spinner="Loading card transactions…")
def load_transactions() -> pd.DataFrame:
    df = _read("card_transactions.csv", parse_dates=[TXN_DATE_COL])
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["score_band"] = pd.cut(
        df["credit_score"], bins=SCORE_BINS, labels=SCORE_BANDS, right=False
    )
    return df


def _read(filename: str, **kwargs) -> pd.DataFrame:
    """Read from Postgres if LENDSCOPE_DB_URL is set, else from CSV."""
    db_url = os.getenv("LENDSCOPE_DB_URL")
    if db_url:
        from sqlalchemy import create_engine

        table = filename.replace(".csv", "")
        engine = _engine(db_url)
        return pd.read_sql(f"SELECT * FROM {table}", engine, **kwargs)

    path = DATA_DIR / filename
    if not path.exists():
        st.error(
            f"Missing `{path}`. Run `python make_sample_data.py` to generate demo data, "
            "or place your real CSVs in the `data/` folder."
        )
        st.stop()
    return pd.read_csv(path, **kwargs)


@st.cache_resource
def _engine(db_url: str):
    from sqlalchemy import create_engine

    return create_engine(db_url, pool_pre_ping=True)


# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------
def apply_filters(
    customers: pd.DataFrame,
    loans: pd.DataFrame,
    txns: pd.DataFrame,
    filters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply sidebar filters. Loan/txn filters cascade from the customer filter."""

    # --- Customers ---
    c = customers.copy()
    if filters["states"]:
        c = c[c["address_state"].isin(filters["states"])]
    if filters["score_range"]:
        lo, hi = filters["score_range"]
        c = c[c["credit_score"].between(lo, hi)]
    if filters["ownership"]:
        c = c[c["home_ownership"].isin(filters["ownership"])]

    # --- Loans ---
    l = loans[loans["customer_id"].isin(c["customer_id"])]
    if filters["grades"]:
        l = l[l["grade"].isin(filters["grades"])]
    if filters["loan_statuses"]:
        l = l[l["loan_status"].isin(filters["loan_statuses"])]
    if filters["purposes"]:
        l = l[l["purpose"].isin(filters["purposes"])]
    if filters["date_range"]:
        start, end = filters["date_range"]
        l = l[l[LOAN_DATE_COL].between(pd.Timestamp(start), pd.Timestamp(end))]

    # --- Transactions ---
    t = txns[txns["customer_id"].isin(c["customer_id"])]
    if filters["card_types"]:
        t = t[t["card_type"].isin(filters["card_types"])]
    if filters["channels"]:
        t = t[t["channel"].isin(filters["channels"])]
    if filters["date_range"]:
        start, end = filters["date_range"]
        t = t[t[TXN_DATE_COL].between(pd.Timestamp(start), pd.Timestamp(end))]

    return c, l, t


def filter_options(loans: pd.DataFrame, txns: pd.DataFrame) -> dict:
    """Distinct values used to populate sidebar widgets."""
    return {
        "grades": sorted(loans["grade"].dropna().unique()),
        "loan_statuses": sorted(loans["loan_status"].dropna().unique()),
        "purposes": sorted(loans["purpose"].dropna().unique()),
        "card_types": sorted(txns["card_type"].dropna().unique()),
        "channels": sorted(txns["channel"].dropna().unique()),
        "ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    }


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def money(x: float) -> str:
    """$1.23M / $45.6K / $789"""
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x/1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:,.1f}K"
    return f"${x:,.0f}"