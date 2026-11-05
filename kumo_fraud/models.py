"""Model wrappers. Every predict function returns P(fraud) as a 1-d float array."""

import time
import warnings
from contextlib import nullcontext

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0


def logreg_fit_predict(X_tr: pd.DataFrame, y_tr, X_te: pd.DataFrame) -> np.ndarray:
    cats = X_tr.select_dtypes("category").columns.tolist()
    nums = [c for c in X_tr.columns if c not in cats]
    pipe = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), cats), (StandardScaler(), nums)
        ),
        LogisticRegression(max_iter=2_000, C=1.0),
    )
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def load_kumo(size: str = "medium", device: str | None = None, pretrained: bool = True):
    import sdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return sdm.models.KumoTabular(
        task="classification", size=size, device=device, pretrained=pretrained
    )


def kumo_predict(
    model,
    X_ctx: pd.DataFrame,
    y_ctx: np.ndarray,
    X_query: pd.DataFrame,
    num_estimators: int = 8,
    query_batch: int = 4_096,
    seed: int = 0,
) -> np.ndarray:
    """In-context prediction: no training, the labeled rows are the prompt. Returns P(fraud)."""
    import sdm

    device = next(model.parameters()).device
    # Kumo only takes numerical features; categoricals go in as category codes.
    df = pd.concat([X_ctx, X_query], ignore_index=True)
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].cat.codes.astype(float)
    df = df.astype(float)
    df["target"] = np.r_[np.asarray(y_ctx).astype(str), np.full(len(X_query), None)]
    table = sdm.TableTensor.from_pandas(
        df=df,
        stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
        device=device,
    )
    n = len(X_ctx)
    ctx, qry = table[:n], table[n:]

    amp = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    gen = torch.Generator(device=device).manual_seed(seed)

    out = []
    with torch.inference_mode(), amp:
        model.fit(
            x=ctx.drop_columns("target"),
            y=ctx[:, "target"],
            num_estimators=num_estimators,
            generator=gen,
        )
        for i in range(0, len(X_query), query_batch):
            pred = model.predict(
                qry[i : i + query_batch].drop_columns("target")
            ).to_pandas()
            out.append(_positive_proba(pred))
        model.clear()
    return np.concatenate(out)


def _positive_proba(pred: pd.DataFrame) -> np.ndarray:
    """Pick the P(class '1') column from Kumo's output frame."""
    for c in pred.columns:
        if str(c).split("_")[-1].split("=")[-1] in ("1", "True"):
            return pred[c].to_numpy(dtype=np.float32)
    raise KeyError(f"no positive-class column in Kumo output: {list(pred.columns)}")


# --------------------------------------------------------------------------- LightGBM
BASE = dict(objective="binary", metric="auc", verbosity=-1, n_jobs=-1)
NO_PREFILTER = {
    "feature_pre_filter": False
}  # lets Optuna vary min_child_samples on a reused Dataset


def lgb_default_fit_predict(X_tr, y_tr, X_te, seed: int = 0) -> np.ndarray:
    """Out-of-the-box LGBMClassifier"""
    m = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def search_space(trial: optuna.Trial) -> dict:
    """Ranges centred on what worked for S6E11: shallow trees, strong regularisation,
    mild positive weighting (public notebooks land around max_depth 3-6)."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
        min_sum_hessian_in_leaf=trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-3, 10, log=True
        ),
        feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 8.0, log=True),
        cat_smooth=trial.suggest_float("cat_smooth", 1, 100, log=True),
        min_data_per_group=trial.suggest_int("min_data_per_group", 10, 300, log=True),
    )


def _folds(y, n_folds, seed):
    return list(
        StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )


def tune_lgb(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 60,
    n_folds: int = 5,
    timeout: float | None = None,
    max_rounds: int = 5_000,
    seed: int = 0,
    study_name: str | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """Optuna (TPE + median pruning) on stratified K-fold ROC-AUC.

    Only ever pass training data here — the holdout must stay untouched. Pass
    storage="sqlite:///../results/optuna.db" to resume / inspect with optuna-dashboard.
    """
    folds = _folds(y, n_folds, seed)
    dsets = [
        (
            lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=False, params=NO_PREFILTER),
            lgb.Dataset(X.iloc[va], y[va], free_raw_data=False, params=NO_PREFILTER),
            y[va],
            X.iloc[va],
        )
        for tr, va in folds
    ]

    def objective(trial):
        params = {**BASE, **search_space(trial), "seed": seed}
        aucs, iters = [], []
        for k, (dtr, dva, yva, Xva) in enumerate(dsets):
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=max_rounds,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            aucs.append(
                roc_auc_score(
                    yva, booster.predict(Xva, num_iteration=booster.best_iteration)
                )
            )
            iters.append(booster.best_iteration)
            trial.report(float(np.mean(aucs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_rounds", int(np.mean(iters)))
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, multivariate=True, n_startup_trials=15
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )
    return study


def best_params(study: optuna.Study) -> tuple[dict, int]:
    t = study.best_trial
    return {**BASE, **t.params, "bagging_freq": 1}, t.user_attrs["n_rounds"]


def lgb_cv_fit_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    X_tests: list[pd.DataFrame],
    params: dict,
    n_folds: int = 5,
    max_rounds: int = 10_000,
    seeds: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """K-fold bagging: returns OOF predictions on X and fold-averaged predictions on each X_test."""
    oof = np.zeros(len(y))
    tests = [np.zeros(len(t)) for t in X_tests]
    for s in seeds:
        for tr, va in _folds(y, n_folds, s):
            booster = lgb.train(
                {**params, "seed": s},
                lgb.Dataset(X.iloc[tr], y[tr]),
                num_boost_round=max_rounds,
                valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            it = booster.best_iteration
            oof[va] += booster.predict(X.iloc[va], num_iteration=it) / len(seeds)
            for i, t in enumerate(X_tests):
                tests[i] += booster.predict(t, num_iteration=it) / (
                    n_folds * len(seeds)
                )
    return oof, tests
