"""Metrik evaluasi, threshold berbasis FPR, dan uji McNemar mid-p."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import binom
from sklearn.metrics import confusion_matrix, roc_auc_score

MAX_FPR: float = 0.01


def threshold_at_fpr(y_val: np.ndarray, p_val: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    """t* = min{ t : FPR_val(t) <= max_fpr }, keputusan AI jika p >= t."""
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    neg_scores = p_val[y_val == 0]
    allowed_fp = int(np.floor(max_fpr * len(neg_scores)))
    for t in np.unique(p_val):
        if (neg_scores >= t).sum() <= allowed_fp:
            return float(t)
    return float(np.nextafter(p_val.max(), np.inf))


def classification_metrics(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
    """Accuracy, precision, recall, F1-score, FPR, FNR, dan AUROC pada threshold t."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
           "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
           "f1": f1, "fpr": fp / (fp + tn) if fp + tn else np.nan,
           "fnr": fn / (fn + tp) if fn + tp else np.nan}
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return out


def mcnemar_midp(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Uji McNemar mid-p. n01 = A salah & B benar; n10 = A benar & B salah."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        p_value = 1.0
    else:
        m = min(n01, n10)
        p_value = float(2 * binom.cdf(m, n, 0.5) - binom.pmf(m, n, 0.5))
        p_value = min(max(p_value, 0.0), 1.0)
    return {"n01": n01, "n10": n10, "n": n, "p_mid": p_value}
