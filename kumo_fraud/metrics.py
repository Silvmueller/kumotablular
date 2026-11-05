import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def recall_at_fpr(y, p, fpr_max: float = 0.01) -> float:
    """Share of frauds caught when at most `fpr_max` of legit transactions are flagged."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr[fpr <= fpr_max]))


def scores(y, p) -> dict:
    """ROC-AUC is the competition metric; the rest matter for an imbalanced real-world use."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "recall@1%FPR": recall_at_fpr(y, p, 0.01),
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def reliability(y, p, n_bins: int = 10):
    """Mean predicted vs. observed fraud rate in quantile bins of p."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    pred = np.array([p[b == i].mean() for i in range(len(edges) - 1)])
    obs = np.array([y[b == i].mean() for i in range(len(edges) - 1)])
    return pred, obs
