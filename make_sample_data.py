"""Generate synthetic LendScope data so the dashboard runs out of the box."""
from pathlib import Path
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

OUT = Path("data")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

STATES = ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI",
          "NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI"]
STATE_P = np.array([12,9,8,7,5,4,4,4,4,3.5,3.5,3.5,3,3,3,3,2.5,2.5,2.5,2.5])
STATE_P = STATE_P / STATE_P.sum()

GRADES   = ["A","B","C","D","E","F","G"]
GRADE_P  = [0.18, 0.28, 0.25, 0.15, 0.08, 0.04, 0.02]
GRADE_INT = {"A":6.3,"B":9.5,"C":12.4,"D":15.6,"E":18.5,"F":21.9,"G":24.5}
GRADE_CO  = {"A":.031,"B":.075,"C":.130,"D":.190,"E":.240,"F":.330,"G":.403}

PURPOSES = ["debt_consolidation","credit_card","home_improvement","major_purchase",
            "medical","small_business","car","vacation","moving","other"]
PURPOSE_P = [0.34,0.22,0.11,0.09,0.06,0.05,0.05,0.04,0.02,0.02]

OWNERSHIP = ["RENT","MORTGAGE","OWN","OTHER"]
OWN_P = [0.42,0.44,0.11,0.03]


def build_customers(n=15000):
    score = np.clip(RNG.normal(700, 75, n), 300, 850).astype(int)
    return pd.DataFrame({
        "customer_id":       np.arange(1, n + 1),
        "first_name":        RNG.choice(["Aisha","Rahul","Maya","Dev","Chen","Sara","Liam","Nia","Omar","Elena"], n),
        "last_name":         RNG.choice(["White","Jain","Patel","Kim","Garcia","Okafor","Nguyen","Silva","Brown","Lopez"], n),
        "address_state":     RNG.choice(STATES, n, p=STATE_P),
        "credit_score":      score,
        "employment_length": np.clip(RNG.normal(6, 4, n), 0, 30).round(1),
        "home_ownership":    RNG.choice(OWNERSHIP, n, p=OWN_P),
        "annual_income":     np.round(np.clip(RNG.lognormal(11.1, 0.5, n), 18000, 500000), -2),
        "dti":               np.clip(RNG.normal(18, 8, n), 0, 60).round(2),
        "age":               np.clip(RNG.normal(42, 12, n), 18, 85).astype(int),
        "emp_title":         RNG.choice(["Engineer","Teacher","Nurse","Analyst","Manager","Driver","Chef","Clerk"], n),
        "email_domain":      RNG.choice(["gmail.com","yahoo.com","outlook.com","proton.me"], n),
        "signup_date":       pd.to_datetime("2024-01-01") + pd.to_timedelta(RNG.integers(0, 730, n), "D"),
    })


def build_loans(customers, n=25000):
    cust = customers.sample(n, replace=True, random_state=1).reset_index(drop=True)
    grade = RNG.choice(GRADES, n, p=GRADE_P)

    # Q4-weighted issue dates within FY2026
    month_w = np.array([6,6,7,7,8,8,9,9,10,10,12,17], dtype=float)
    month_w /= month_w.sum()
    months = RNG.choice(np.arange(1, 13), n, p=month_w)
    days = RNG.integers(1, 28, n)
    issue_date = pd.to_datetime(dict(year=2026, month=months, day=days))

    funded = np.round(np.clip(RNG.lognormal(9.7, 0.7, n), 1000, 40000), -2)
    int_rate = np.array([GRADE_INT[g] for g in grade]) + RNG.normal(0, 1.2, n)
    term = RNG.choice([36, 60], n, p=[0.55, 0.45])

    co_p = np.array([GRADE_CO[g] for g in grade])
    is_charged_off = RNG.random(n) < co_p

    status = np.where(
        is_charged_off,
        "Charged Off",
        RNG.choice(["Fully Paid", "Current"], n, p=[0.30, 0.70]),
    )

    # Payment received depends on status and elapsed time
    elapsed = (pd.Timestamp("2027-01-01") - issue_date).dt.days / 365.0
    fully_paid_frac = 1.0 + (int_rate / 100) * (term / 12)
    current_frac = np.clip(elapsed * (12 / term), 0.02, 0.95)
    co_frac = np.clip(elapsed * 0.12, 0.02, 0.45)

    total_payment = np.where(
        status == "Fully Paid", funded * fully_paid_frac,
        np.where(status == "Current", funded * current_frac, funded * co_frac),
    )

    return pd.DataFrame({
        "loan_id":        np.arange(1, n + 1),
        "customer_id":    cust["customer_id"].values,
        "issue_date":     issue_date,
        "funded_amount":  funded,
        "total_payment":  np.round(total_payment, 2),
        "int_rate":       np.round(int_rate, 2),
        "grade":          grade,
        "term_months":    term,
        "loan_status":    status,
        "purpose":        RNG.choice(PURPOSES, n, p=PURPOSE_P),
        "dti":            np.clip(cust["dti"].values + RNG.normal(0, 3, n), 0, 60).round(2),
        "annual_income":  cust["annual_income"].values,
        "address_state":  cust["address_state"].values,
        "credit_score":   cust["credit_score"].values,
        "home_ownership": cust["home_ownership"].values,
        "employment_length": cust["employment_length"].values,
        "verification_status": RNG.choice(["Verified","Source Verified","Not Verified"], n, p=[.4,.35,.25]),
        "application_type": RNG.choice(["Individual","Joint"], n, p=[.88,.12]),
        "delinq_2yrs":    RNG.poisson(0.3, n),
        "open_accounts":  RNG.integers(2, 25, n),
        "revol_util":     np.clip(RNG.normal(48, 24, n), 0, 120).round(1),
        "installment":    np.round(funded / term * (1 + int_rate / 100), 2),
        "emp_length_yrs": cust["employment_length"].values,
        "sub_grade":      [f"{g}{RNG.integers(1,6)}" for g in grade],
        "desc_len":       RNG.integers(0, 400, n),
    })


def build_transactions(customers, n=75000):
    cust = customers.sample(n, replace=True, random_state=2).reset_index(drop=True)
    card_type = RNG.choice(["credit", "debit"], n, p=[0.55, 0.45])
    channel = RNG.choice(["POS", "ONLINE", "ATM"], n, p=[0.62, 0.28, 0.10])

    # Fraud probability: ONLINE >> POS ≈ ATM  (note: "is_fraud" not "is_fraud_transaction")
    base = np.where(channel == "ONLINE", 0.0081, np.where(channel == "ATM", 0.0027, 0.0026))
    is_fraud = RNG.random(n) < base

    # Q4 spend weighting
    month_w = np.array([7,6,7,7,7,8,8,8,9,9,11,13], dtype=float)
    month_w /= month_w.sum()
    months = RNG.choice(np.arange(1, 13), n, p=month_w)
    tx_date = pd.to_datetime(dict(year=2026, month=months, day=RNG.integers(1, 28, n)))

    amount = np.round(np.clip(RNG.lognormal(3.6, 1.1, n), 1, 5000), 2)
    amount = np.where(is_fraud, amount * RNG.uniform(1.5, 4.0, n), amount).round(2)

    return pd.DataFrame({
        "transaction_id":   np.arange(1, n + 1),
        "customer_id":      cust["customer_id"].values,
        "transaction_date": tx_date,
        "amount":           amount,
        "card_type":        card_type,
        "channel":          channel,
        "merchant_category": RNG.choice(["grocery","travel","electronics","fuel","dining","retail","utilities"], n),
        "is_fraud":         is_fraud.astype(int),
        "fraud_amount":     np.where(is_fraud, amount, 0.0).round(2),
        "address_state":    cust["address_state"].values,
        "credit_score":     cust["credit_score"].values,
        "card_brand":       RNG.choice(["visa","mastercard","amex","discover"], n, p=[.45,.35,.12,.08]),
        "auth_method":      RNG.choice(["chip","contactless","swipe","online"], n),
        "currency":         "USD",
    })


if __name__ == "__main__":
    customers = build_customers()
    loans = build_loans(customers)
    txns = build_transactions(customers)

    customers.to_csv(OUT / "customers.csv", index=False)
    loans.to_csv(OUT / "loans.csv", index=False)
    txns.to_csv(OUT / "card_transactions.csv", index=False)
    print(f"Wrote {len(customers):,} customers | {len(loans):,} loans | {len(txns):,} transactions")