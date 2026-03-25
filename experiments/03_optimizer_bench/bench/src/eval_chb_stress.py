from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import auc_roc


def add_noise_to_snr(x: np.ndarray, snr_db: float, eps: float = 1e-12) -> np.ndarray:
    """x: [C, T]"""
    sig_power = np.mean(x ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_lin + eps)
    noise = np.random.randn(*x.shape).astype(x.dtype) * np.sqrt(noise_power)
    return x + noise


def dropout_channels(x: np.ndarray, k: int) -> np.ndarray:
    """Zero out k random channels."""
    c = x.shape[0]
    k = min(k, c)
    idx = np.random.choice(c, size=k, replace=False)
    x2 = x.copy()
    x2[idx] = 0.0
    return x2


@torch.no_grad()
def eval_auc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    ys = []
    ps = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        p = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)
    return float(auc_roc(y_true, y_score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to torch state_dict")
    ap.add_argument("--in_metrics_json", type=str, required=True, help="Existing run metrics.json")
    ap.add_argument("--out_metrics_json", type=str, required=True, help="Write updated metrics.json")
    ap.add_argument("--noise_snr_db", type=float, nargs="+", default=[20, 10])
    ap.add_argument("--channel_dropout_k", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load metrics json to update
    with open(args.in_metrics_json, "r") as f:
        payload = json.load(f)

    # You must implement a way to reconstruct your CHB test loader here.
    # This script assumes you already have a serialized test set of windows.
    # Recommended: save numpy arrays for test windows alongside run outputs.
    raise NotImplementedError(
        "Wire eval_chb_stress.py to your CHB test loader. "
        "Then compute AUC under noise/dropout and update payload['robust_scores']."
    )


if __name__ == "__main__":
    main()