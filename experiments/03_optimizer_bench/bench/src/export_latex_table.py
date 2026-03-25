from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sf(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def fmt_mean_std(mean: Optional[float], std: Optional[float], *, decimals: int = 3) -> str:
    m = _sf(mean)
    s = _sf(std)
    if m is None:
        return "--"
    if s is None:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} $\pm$ {s:.{decimals}f}"


def fmt_time(mean: Optional[float], std: Optional[float]) -> str:
    m = _sf(mean)
    s = _sf(std)
    if m is None:
        return "--"
    if s is None:
        return f"{m:.0f}"
    return f"{m:.0f} $\pm$ {s:.0f}"


def choose_best_by_iid(rows: List[Dict[str, Any]], domain: str) -> Dict[str, Dict[str, Any]]:
    """Return mapping optimizer -> best row for that optimizer, choosing the row with max iid_mean.

    This merges regimes (fixed_time/fixed_compute) in a deterministic way.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("domain") != domain:
            continue
        opt = str(r.get("optimizer"))
        iid = _sf(r.get("iid_mean"))
        if iid is None:
            continue
        if opt not in best or iid > float(best[opt].get("iid_mean", -1e9)):
            best[opt] = r
    return best


def extract_efficiency(r: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Preferred efficiency column: time_to_target; fallback: budget wallclock."""
    m = _sf(r.get("time_to_target_s_mean"))
    s = _sf(r.get("time_to_target_s_std"))
    if m is not None:
        return m, s
    return _sf(r.get("budget_wallclock_seconds_mean")), _sf(r.get("budget_wallclock_seconds_std"))


def pretty_optimizer_name(opt: str) -> str:
    """Paper-facing optimizer names (LaTeX-safe; matches the table style in the manuscript)."""
    key = (opt or "").lower()

    mapping = {
        "sgd": r"SGD",
        "sgd_momentum": r"SGD\texttt{+}Momentum",
        "adam": r"Adam",
        "adamw": r"AdamW",
        "rmsprop": r"RMSProp",
        "lion": r"Lion",
        "sam_sgd": r"SAM (SGD)",
        "sam_adamw": r"SAM (AdamW)",
    }

    name = mapping.get(key, opt)

    # Make LaTeX-safe (keep + handling in mapping above)
    name = str(name).replace("_", r"\\_")
    return name


def default_caption() -> str:
    # NeurIPS-style exemplar caption: short, specific, and describes aggregation + bolding.
    return (
        r"\textbf{End-to-end optimizer benchmark on CIFAR-10 and CHB-MIT.} "
        "We report mean $\pm$ standard deviation over three random seeds for IID test accuracy (CIFAR-10) "
        "and Independent and Identically Distributed (IID) test AUC (CHB-MIT), alongside time-to-target in seconds (lower is better; falling back to the fixed budget when time-to-target is unavailable). "
        "Best performance (higher is better) and best time (lower is better) are highlighted in bold for each dataset."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", type=str, default="bench/results/aggregated/summary.json")
    ap.add_argument(
        "--optimizers",
        type=str,
        default="sgd,sgd_momentum,adam,adamw,rmsprop,lion,sam_sgd,sam_adamw",
        help="Comma-separated optimizer order for the table",
    )
    ap.add_argument("--label", type=str, default="tab:results")
    ap.add_argument("--caption", type=str, default=None)
    args = ap.parse_args()

    caption = default_caption() if not args.caption else str(args.caption)

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    rows = summary.get("groups", [])
    if not isinstance(rows, list):
        raise ValueError("summary['groups'] must be a list")

    opt_order = [o.strip() for o in str(args.optimizers).split(",") if o.strip()]

    best_cifar = choose_best_by_iid(rows, "cifar10")
    best_chb = choose_best_by_iid(rows, "chb")

    # Identify best values for boldface (per domain)
    cif_accs = []
    cif_times = []
    for r in best_cifar.values():
        a = _sf(r.get("iid_mean"))
        if a is not None:
            cif_accs.append(a)
        tm, _ts = extract_efficiency(r)
        if tm is not None:
            cif_times.append(tm)

    chb_aucs = []
    chb_times = []
    for r in best_chb.values():
        a = _sf(r.get("iid_mean"))
        if a is not None:
            chb_aucs.append(a)
        tm, _ts = extract_efficiency(r)
        if tm is not None:
            chb_times.append(tm)

    best_cif_acc = float(np.max(cif_accs)) if cif_accs else None
    best_cif_time = float(np.min(cif_times)) if cif_times else None
    best_chb_auc = float(np.max(chb_aucs)) if chb_aucs else None
    best_chb_time = float(np.min(chb_times)) if chb_times else None

    # Full NeurIPS-style table wrapper
    print(r"\begin{table}[htp!]")
    print(r"\centering")

    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{CHB-MIT} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r"Optimizer & Acc. (\%) & Time (s) & AUC (\%) & Time (s) \\")
    print(r"\midrule")

    def row_for(opt: str) -> str:
        rc = best_cifar.get(opt)
        rh = best_chb.get(opt)

        # CIFAR
        if rc is None:
            cif_acc = "--"
            cif_t = "--"
        else:
            cif_mean = _sf(rc.get("iid_mean"))
            cif_std = _sf(rc.get("iid_std"))
            cif_acc = fmt_mean_std(
                100.0 * cif_mean if cif_mean is not None else None,
                100.0 * cif_std if cif_std is not None else None,
                decimals=2,
            )

            t_m, t_s = extract_efficiency(rc)
            cif_t = fmt_time(t_m, t_s)

            if best_cif_acc is not None and cif_mean is not None and abs(cif_mean - best_cif_acc) <= 1e-12:
                cif_acc = r"\textbf{" + cif_acc + "}"
            if best_cif_time is not None and t_m is not None and abs(t_m - best_cif_time) <= 1e-9:
                cif_t = r"\textbf{" + cif_t + "}"

        # CHB
        if rh is None:
            chb_auc = "--"
            chb_t = "--"
        else:
            chb_mean = _sf(rh.get("iid_mean"))
            chb_std = _sf(rh.get("iid_std"))
            chb_auc = fmt_mean_std(
                100.0 * chb_mean if chb_mean is not None else None,
                100.0 * chb_std if chb_std is not None else None,
                decimals=2,
            )

            t_m, t_s = extract_efficiency(rh)
            chb_t = fmt_time(t_m, t_s)

            if best_chb_auc is not None and chb_mean is not None and abs(chb_mean - best_chb_auc) <= 1e-12:
                chb_auc = r"\textbf{" + chb_auc + "}"
            if best_chb_time is not None and t_m is not None and abs(t_m - best_chb_time) <= 1e-9:
                chb_t = r"\textbf{" + chb_t + "}"

        opt_tex = pretty_optimizer_name(opt)
        return f"{opt_tex} & {cif_acc} & {cif_t} & {chb_auc} & {chb_t} \\\\ "

    for opt in opt_order:
        print(row_for(opt))

    print(r"\bottomrule")
    print(r"\end{tabular}")

    print(r"\caption{" + caption + r"}")
    print(r"\label{" + str(args.label) + r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()