from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Simple rankdata without ties handling sophistication.

    For AUC, ties exist; for robust use, prefer sklearn. This is a fallback.
    """
    temp = a.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a), dtype=float)
    return ranks


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC.

    Prefers sklearn if available, else uses a rank-based implementation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Mann–Whitney U statistic / rank method
        pos = y_true == 1
        neg = y_true == 0
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = _rankdata(y_score)
        sum_ranks_pos = float(ranks[pos].sum())
        # U = sum_ranks_pos - n_pos*(n_pos-1)/2
        u = sum_ranks_pos - n_pos * (n_pos - 1) / 2.0
        return float(u / (n_pos * n_neg))


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true))


def bce_logits_to_scores(logits: np.ndarray) -> np.ndarray:
    """Binary classification: logits shape [N] or [N,1]"""
    logits = np.asarray(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    return sigmoid(logits)


def multiclass_logits_to_scores(logits: np.ndarray) -> np.ndarray:
    """Multiclass probabilities"""
    return softmax(np.asarray(logits), axis=1)


def stable_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    return float(np.mean(xs))


def stable_min(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    return float(np.min(xs))