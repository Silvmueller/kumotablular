"""Produce out-of-fold predictions for every model on the GPU machine.

Outer loop: stratified K-fold over all 700k training rows. In fold k each model
learns from N rows drawn from the other folds and predicts the whole held-out
fold (~140k rows, ~1.9k frauds), so every (model, N) gets an OOF prediction for
every training row. Optuna tunes LightGBM inside the training rows only (nested).

Outputs (read by notebooks/analysis.ipynb, no GPU needed there):
  results/oof/n{N}.csv.gz      id, fold, is_fraud, one column per model
  results/oof/full.csv.gz      same, LightGBM on all fold-train rows, Kumo on its largest context
  results/test/full.csv.gz     id + test predictions of the full setting (fold-averaged)
  results/timings.csv          n, fold, model, seconds
  results/params.json          best Optuna params per (n, fold)

Finished folds are cached in results/oof/parts/, so an interrupted run resumes.

    python run_oof.py                          # everything
    python run_oof.py --sizes 16000 --no-full  # one setting only
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kumo_fraud.data import ID, TARGET, add_features, load, sample, xy
from kumo_fraud.metrics import scores
from kumo_fraud.models import (
    Timer,
    best_params,
    kumo_predict,
    lgb_cv_fit_predict,
    lgb_default_fit_predict,
    load_kumo,
    logreg_fit_predict,
    tune_lgb,
)

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--original", default=None, help="original 20k-row dataset to append (optional)")
p.add_argument("--out", default="results")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--sizes", type=int, nargs="+", default=[1_000, 4_000, 16_000, 64_000])
p.add_argument("--trials", type=int, default=40, help="Optuna trials per (N, fold)")
p.add_argument("--pos-fracs", type=float, nargs="*", default=[0.05, 0.10, 0.25],
               help="extra Kumo runs with fraud-enriched context, at --enrich-n")
p.add_argument("--enrich-n", type=int, default=16_000)
p.add_argument("--no-full", action="store_true", help="skip the full-data setting")
p.add_argument("--tune-rows", type=int, default=150_000, help="Optuna subsample in the full setting")
p.add_argument("--full-trials", type=int, default=60)
p.add_argument("--kumo-size", default="medium")
p.add_argument("--kumo-estimators", type=int, default=8)
p.add_argument("--kumo-context", type=int, default=50_000, help="Kumo context in the full setting")
p.add_argument("--kumo-pos-frac", type=float, default=None, help="enrichment in the full setting")
p.add_argument("--no-kumo", action="store_true")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

out = Path(a.out)
parts = out / "oof" / "parts"
for d in (parts, out / "test"):
    d.mkdir(parents=True, exist_ok=True)
storage = f"sqlite:///{out / 'optuna.db'}"

train, test = load(a.data_dir, original=a.original)
train, test = add_features(train), add_features(test)
X_all, y_all = xy(train)
X_test, _ = xy(test)
folds = list(StratifiedKFold(a.folds, shuffle=True, random_state=a.seed).split(X_all, y_all))
print(f"train {len(train):,} | test {len(test):,} | fraud {y_all.mean():.4f} | {a.folds} folds")

kumo = None
if not a.no_kumo:
    kumo = load_kumo(a.kumo_size)
    _ = kumo_predict(kumo, X_all[:200], y_all[:200], X_all[:10], num_estimators=1)  # warm-up

params_path = out / "params.json"
all_params = json.loads(params_path.read_text()) if params_path.exists() else {}
timings_path = out / "timings.csv"


def log_time(n, k, model, sec):
    row = pd.DataFrame([{"n": n, "fold": k, "model": model, "seconds": sec}])
    row.to_csv(timings_path, mode="a", header=not timings_path.exists(), index=False)


def run_fold(n, k, tr_idx, va_idx):
    """Train every model on N rows from tr_idx, predict va_idx. Returns OOF frame (+ test preds)."""
    tag = "full" if n is None else f"n{n}"
    pool = train.iloc[tr_idx].reset_index(drop=True)
    X_va = X_all.iloc[va_idx]
    seed = a.seed + k
    ctx = pool if n is None else sample(pool, n, seed=seed)
    X_ctx, y_ctx = xy(ctx)
    res = {ID: train[ID].iloc[va_idx].to_numpy(), "fold": k, TARGET: y_all[va_idx]}
    test_preds = {}

    def timed(name, fn):
        with Timer() as t:
            out_ = fn()
        log_time(tag, k, name, t.seconds)
        return out_

    if n is not None:
        res["logreg"] = timed("logreg", lambda: logreg_fit_predict(X_ctx, y_ctx, X_va))
        res["lgb_default"] = timed("lgb_default", lambda: lgb_default_fit_predict(X_ctx, y_ctx, X_va, seed=seed))

    # LightGBM + Optuna (nested: tuned on training rows of this fold only)
    def lgb_tuned():
        X_tune, y_tune = (X_ctx, y_ctx) if n is not None else xy(sample(pool, a.tune_rows, seed=seed))
        study = tune_lgb(X_tune, y_tune, n_trials=a.trials if n is not None else a.full_trials,
                         seed=seed, study_name=f"{tag}_fold{k}", storage=storage)
        params, _ = best_params(study)
        all_params[f"{tag}_fold{k}"] = {"cv_auc": study.best_value, **params}
        params_path.write_text(json.dumps(all_params, indent=1))
        tests = [X_va, X_test] if n is None else [X_va]
        _, preds = lgb_cv_fit_predict(X_ctx, y_ctx, tests, params, seeds=(seed,))
        return preds

    preds = timed("lgb_optuna", lgb_tuned)
    res["lgb_optuna"] = preds[0]
    if n is None:
        test_preds["lgb_optuna"] = preds[1]

    if kumo is not None:
        if n is None:
            kctx = sample(pool, a.kumo_context, pos_frac=a.kumo_pos_frac, seed=seed)
            q = pd.concat([X_va, X_test], ignore_index=True)
            pk = timed("kumo", lambda: kumo_predict(kumo, *xy(kctx), q,
                                                     num_estimators=a.kumo_estimators, seed=seed))
            res["kumo"], test_preds["kumo"] = pk[: len(X_va)], pk[len(X_va):]
        else:
            res["kumo"] = timed("kumo", lambda: kumo_predict(kumo, X_ctx, y_ctx, X_va,
                                                              num_estimators=a.kumo_estimators, seed=seed))
            if n == a.enrich_n:
                for pf in a.pos_fracs:
                    c = sample(pool, n, pos_frac=pf, seed=seed)
                    res[f"kumo_pos{pf:.2f}"] = timed(f"kumo_pos{pf:.2f}", lambda: kumo_predict(
                        kumo, *xy(c), X_va, num_estimators=a.kumo_estimators, seed=seed))

    return pd.DataFrame(res), test_preds


settings = list(a.sizes) + ([] if a.no_full else [None])
for n in settings:
    tag = "full" if n is None else f"n{n}"
    frames, test_acc = [], {}
    for k, (tr_idx, va_idx) in enumerate(folds):
        part = parts / f"{tag}_fold{k}.csv.gz"
        tpart = parts / f"{tag}_fold{k}_test.csv.gz"
        if part.exists():
            df = pd.read_csv(part)
            tp = pd.read_csv(tpart).to_dict("series") if tpart.exists() else {}
        else:
            df, tp = run_fold(n, k, tr_idx, va_idx)
            df.to_csv(part, index=False)
            if tp:
                pd.DataFrame(tp).to_csv(tpart, index=False)
        frames.append(df)
        for m, v in tp.items():
            test_acc[m] = test_acc.get(m, 0) + np.asarray(v) / len(folds)
        cols = [c for c in df.columns if c not in (ID, "fold", TARGET)]
        print(f"{tag} fold {k}: " + " | ".join(
            f"{c} {scores(df[TARGET].to_numpy(), df[c].to_numpy())['ROC_AUC']:.4f}" for c in cols))

    oof = pd.concat(frames).sort_values(ID)
    oof.to_csv(out / "oof" / f"{tag}.csv.gz", index=False)
    if test_acc:
        pd.DataFrame({ID: test[ID], **test_acc}).to_csv(out / "test" / f"{tag}.csv.gz", index=False)

print("done ->", out.resolve())
