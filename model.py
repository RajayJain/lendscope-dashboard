"""XGBoost default-prediction model for LendScope.

Trains on the loans table, scores individual applications, and returns
log-odds SHAP contributions using XGBoost's native `pred_contribs`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lendscope_xgb.json"
META_PATH = MODEL_DIR / "lendscope_xgb_meta.joblib"

TARGET = "is_bad"

NUMERIC_FEATURES = [
    "int_rate", "term_months", "dti", "annual_income", "credit_score",
    "employment_length", "delinq_2yrs", "open_accounts", "revol_util",
    "installment", "funded_amount",
]
CATEGORICAL_FEATURES = [
    "grade", "purpose", "home_ownership", "address_state",
    "verification_status", "application_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Reasonable seed values for the interactive scorer.
DEFAULTS: dict = {
    "int_rate": 12.5,
    "term_months": 36,
    "dti": 18.0,
    "annual_income": 65000.0,
    "credit_score": 700,
    "employment_length": 5.0,
    "delinq_2yrs": 0,
    "open_accounts": 10,
    "revol_util": 45.0,
    "installment": 350.0,
    "funded_amount": 12000.0,
    "grade": "C",
    "purpose": "debt_consolidation",
    "home_ownership": "RENT",
    "address_state": "CA",
    "verification_status": "Verified",
    "application_type": "Individual",
}

# Business decision bands (probability of default).
APPROVE_BELOW = 0.10
DECLINE_ABOVE = 0.25


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def _prepare_X(df: pd.DataFrame, cat_categories: dict | None = None) -> pd.DataFrame:
    """Select features, coerce dtypes, and align categoricals to training codes."""
    X = df[FEATURES].copy()
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        s = X[c].astype("string").fillna("Unknown")
        if cat_categories and c in cat_categories:
            X[c] = pd.Categorical(s, categories=cat_categories[c])
        else:
            X[c] = s.astype("category")
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(loans: pd.DataFrame, random_state: int = 42):
    """Train the XGBoost classifier. Returns (model, meta, eval_bundle)."""
    df = loans.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = _prepare_X(df)
    y = df[TARGET]

    num_medians = X[NUMERIC_FEATURES].median()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(num_medians)

    cat_categories = {c: X[c].cat.categories.tolist() for c in CATEGORICAL_FEATURES}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, y_prob))
    ap = float(average_precision_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, thr = precision_recall_curve(y_test, y_prob)

    # F1-optimal threshold on the test set
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    best_idx = int(np.argmax(f1))
    best_thr = float(thr[best_idx]) if len(thr) else 0.5

    y_pred = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    meta = {
        "auc": auc,
        "average_precision": ap,
        "best_threshold": best_thr,
        "positive_rate": pos / max(pos + neg, 1),
        "scale_pos_weight": spw,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "num_medians": {k: float(v) for k, v in num_medians.items()},
        "cat_categories": cat_categories,
        "trained_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "features": FEATURES,
    }

    eval_bundle = {
        "roc": (fpr, tpr),
        "pr": (prec, rec),
        "confusion": cm,
        "threshold": best_thr,
        "y_test": y_test.reset_index(drop=True),
        "y_prob": y_prob,
        "auc": auc,
        "average_precision": ap,
    }
    return model, meta, eval_bundle


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model: xgb.XGBClassifier, meta: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model.save_model(MODEL_PATH)
    joblib.dump(meta, META_PATH)


def load_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = xgb.XGBClassifier(enable_categorical=True, tree_method="hist")
    model.load_model(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


# ---------------------------------------------------------------------------
# Scoring + explanation
# ---------------------------------------------------------------------------
def score_application(
    model: xgb.XGBClassifier, meta: dict, application: dict
) -> dict:
    """Score one application and return probability + SHAP contributions.

    Returns a dict with:
      prob         — probability of default (0–1)
      log_odds     — raw margin (base + sum of contributions)
      base         — model bias (log-odds)
      features     — list of feature names
      contributions— np.ndarray of log-odds contributions (same order)
    """
    row = pd.DataFrame([application])
    X = _prepare_X(row, cat_categories=meta.get("cat_categories"))
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(pd.Series(meta["num_medians"]))

    prob = float(model.predict_proba(X)[:, 1][0])

    booster = model.get_booster()
    dmat = xgb.DMatrix(X, enable_categorical=True)
    contribs = booster.predict(dmat, pred_contribs=True)[0]

    return {
        "prob": prob,
        "log_odds": float(contribs.sum()),
        "base": float(contribs[-1]),
        "features": list(FEATURES),
        "contributions": contribs[:-1],
    }


def global_feature_importance(model: xgb.XGBClassifier) -> pd.DataFrame:
    """Gain-based importance, aligned to FEATURES order."""
    gains = model.feature_importances_
    df = pd.DataFrame({"feature": FEATURES, "gain": gains})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    total = df["gain"].sum()
    df["gain_share"] = df["gain"] / total if total > 0 else 0.0
    return df


def decision_band(prob: float) -> tuple[str, str]:
    """Return (label, hex_color) for a probability of default."""
    if prob < APPROVE_BELOW:
        return "Approve", "#2e7d32"
    if prob < DECLINE_ABOVE:
        return "Manual Review", "#f9a825"
    return "Decline", "#c62828"