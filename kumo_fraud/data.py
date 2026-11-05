"""Load the Kaggle Playground S6E11 fraud data ("Predicting Credit Card Fraud")."""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "is_fraud"
ID = "id"

CATEGORICAL = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLEAN = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def _clean(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    for c in BOOLEAN:
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(np.int8)
    for c in CATEGORICAL:
        if c in df:
            df[c] = (
                pd.Categorical(df[c], categories=categories[c])
                if categories
                else df[c].astype("category")
            )
    return df


def load(data_dir: str | Path = "../data", original: str | Path | None = None):
    """Return (train, test) with identical feature columns and shared category codes.

    `original`: optional path to the source dataset the competition was generated from
    (kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026). Its rows are
    appended to train with `is_original = 1`; only columns shared with train are kept.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    if original is not None:
        orig = pd.read_csv(original)
        orig = orig[[c for c in train.columns if c in orig.columns and c != ID]]
        orig = orig.dropna(subset=[TARGET])
        train = pd.concat([train.assign(is_original=0), orig.assign(is_original=1)], ignore_index=True)
        test = test.assign(is_original=0)

    train[TARGET] = train[TARGET].astype(int)
    full = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    cats = {c: sorted(full[c].dropna().astype(str).unique()) for c in CATEGORICAL}
    return _clean(train, cats), _clean(test, cats)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small set of hand-made features. Applied to every model so the comparison stays fair."""
    df = df.copy()
    df["n_risk_flags"] = df[BOOLEAN].sum(axis=1) + (df["cvv_retry_count"] > 0)
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"].abs() + 1)
    df["log_amount"] = np.log1p(df["amount_usd"].clip(lower=0))
    df["night"] = df["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(np.int8)
    df["no_auth"] = (df["auth_method"].astype(str) == "No Authentication").astype(np.int8)
    return df


def holdout_split(df: pd.DataFrame, frac: float = 0.2, seed: int = 0):
    """Stratified split -> (pool, holdout). Holdout is never used for tuning."""
    rng = np.random.default_rng(seed)
    idx = np.empty(0, dtype=int)
    for _, g in df.groupby(TARGET):
        n = int(round(len(g) * frac))
        idx = np.concatenate([idx, rng.choice(g.index.to_numpy(), n, replace=False)])
    mask = df.index.isin(idx)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def sample(pool: pd.DataFrame, n: int, pos_frac: float | None = None, seed: int = 0) -> pd.DataFrame:
    """Draw n rows. pos_frac=None keeps the natural fraud rate (~1.7 %), otherwise
    positives are enriched to that share (rank metrics like AUC are unaffected by this)."""
    n = min(n, len(pool))
    pos, neg = pool[pool[TARGET] == 1], pool[pool[TARGET] == 0]
    share = len(pos) / len(pool) if pos_frac is None else pos_frac
    n_pos = min(len(pos), int(round(n * share)))
    out = pd.concat(
        [pos.sample(n=n_pos, random_state=seed), neg.sample(n=n - n_pos, random_state=seed)]
    )
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def xy(df: pd.DataFrame):
    X = df.drop(columns=[c for c in (TARGET, ID) if c in df])
    y = df[TARGET].to_numpy(dtype=int) if TARGET in df else None
    return X, y
