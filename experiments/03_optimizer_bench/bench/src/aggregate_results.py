from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def bench_root() -> Path:
    # This file lives at bench/src/aggregate_results.py
    return Path(__file__).resolve().parents[1]


def load_all_metrics(root: str) -> List[Dict[str, Any]]:
    paths = glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True)
    rows = []
    for p in paths:
        try:
            with open(p, "r") as f:
                j = json.load(f)
            j["_path"] = p
            rows.append(j)
        except Exception:
            continue
    return rows


def _finite(xs: List[Optional[float]]) -> List[float]:
    out = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if np.isfinite(xf):
            out.append(xf)
    return out


def main():
    ap = argparse.ArgumentParser()

    # Defaults are rooted at bench/ so running from any working directory is consistent.
    ap.add_argument("--results_root", type=str, default=str(bench_root() / "results" / "raw"))
    ap.add_argument("--out_json", type=str, default=str(bench_root() / "results" / "aggregated" / "summary.json"))
    args = ap.parse_args()

    rows = load_all_metrics(args.results_root)

    # group by (domain, regime, optimizer)
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("domain"), r.get("regime"), r.get("optimizer"))
        groups[key].append(r)

    out: Dict[str, Any] = {"groups": []}

    for (domain, regime, optimizer), rs in sorted(groups.items()):
        # -----------------
        # Success / failure
        # -----------------
        # Mark a run as "success" if it has a finite iid_score.
        iid_scores = _finite([r.get("iid_score") for r in rs])
        n_total = len(rs)
        n_success = len(iid_scores)
        failure_rate = None if n_total == 0 else float(1.0 - (n_success / n_total))

        # -----------------
        # IID
        # -----------------
        iid_mean = float(np.mean(iid_scores)) if iid_scores else None
        iid_std = float(np.std(iid_scores, ddof=1)) if len(iid_scores) >= 2 else None

                # -----------------
        # Time-to-target (fixed target, not relative)
        # -----------------

        # Expected field populated by training scripts:
        # metrics["time_to_target_s"]["acc@0.90"] (cifar) or ["auc@0.90"] (chb), etc.
        ttts = []
        for r in rs:
            tmap = (r.get("time_to_target_s") or {})
            # prefer CIFAR key if present else CHB key
            v = tmap.get("acc@0.90", None)
            if v is None:
                v = tmap.get("auc@0.90", None)
            ttts.append(v)
        ttts = _finite(ttts)
        time_to_target_s_mean = float(np.mean(ttts)) if ttts else None
        time_to_target_s_std = float(np.std(ttts, ddof=1)) if len(ttts) >= 2 else None

        # -----------------
        # Robustness (only if explicitly available)
        # -----------------
        # CIFAR-10-C fields expected under robust_scores.
        cifar10c_mean = []
        cifar10c_worst = []
        chb_stress_mean = []
        chb_stress_worst = []

        for r in rs:
            rob = (r.get("robust_scores") or {})
            cifar10c_mean.append(rob.get("cifar10c_mean"))
            cifar10c_worst.append(rob.get("cifar10c_worst"))

            # CHB stress keys if you later add them
            chb_vals = _finite([rob.get(k) for k in ["snr20db", "snr10db", "drop2ch", "drop4ch"]])
            if chb_vals:
                chb_stress_mean.append(float(np.mean(chb_vals)))
                chb_stress_worst.append(float(np.min(chb_vals)))

        cifar10c_mean = _finite(cifar10c_mean)
        cifar10c_worst = _finite(cifar10c_worst)

        cifar10c_mean_mean = float(np.mean(cifar10c_mean)) if cifar10c_mean else None
        cifar10c_worst_mean = float(np.mean(cifar10c_worst)) if cifar10c_worst else None

        chb_stress_mean_mean = float(np.mean(chb_stress_mean)) if chb_stress_mean else None
        chb_stress_worst_mean = float(np.mean(chb_stress_worst)) if chb_stress_worst else None

        # -----------------
        # Budget passthrough (for plotting fallbacks)
        # -----------------
        # Many analyses want an efficiency-cost axis even when time-to-target is missing.
        # We aggregate budget.wallclock_seconds here so downstream plots can fall back
        # without scraping raw metrics.json files.
        bws = _finite([(r.get("budget") or {}).get("wallclock_seconds") for r in rs])
        budget_wallclock_seconds_mean = float(np.mean(bws)) if bws else None
        budget_wallclock_seconds_std = float(np.std(bws, ddof=1)) if len(bws) >= 2 else None

        out["groups"].append(
            {
                "domain": domain,
                "regime": regime,
                "optimizer": optimizer,
                "n_total": n_total,
                "n_success": n_success,
                "failure_rate": failure_rate,
                "iid_mean": iid_mean,
                "iid_std": iid_std,
                "time_to_target_s_mean": time_to_target_s_mean,
                "time_to_target_s_std": time_to_target_s_std,
                "budget_wallclock_seconds_mean": budget_wallclock_seconds_mean,
                "budget_wallclock_seconds_std": budget_wallclock_seconds_std,
                # CIFAR (optional)
                "cifar10c_mean_mean": cifar10c_mean_mean,
                "cifar10c_worst_mean": cifar10c_worst_mean,
                # CHB (optional)
                "chb_stress_mean_mean": chb_stress_mean_mean,
                "chb_stress_worst_mean": chb_stress_worst_mean,
            }
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()