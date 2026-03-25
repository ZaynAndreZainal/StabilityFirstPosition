from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Keep optimizer colors consistent with plot_pareto_tradeoff.py
OPT_COLORS = {
    "sgd": "#1f77b4",          # blue
    "sgd_momentum": "#17becf", # cyan
    "adam": "#ff7f0e",         # orange
    "adamw": "#d62728",        # red
    "rmsprop": "#8c564b",      # brown
    "lion": "#9467bd",         # purple
    "sam": "#7f7f7f",          # gray (unused; SAM is not tested as a standalone optimizer)
    "sam_sgd": "#1B7F3A",      # dark green (SAM family shade 1)
    "sam_adamw": "#7BC96F",    # light green (SAM family shade 2)
}


def optimizer_color(opt_name: str) -> str:
    if opt_name is None:
        return "#7f7f7f"  # gray
    return OPT_COLORS.get(str(opt_name).lower(), "#7f7f7f")


def load_runs(results_root: str) -> List[Dict[str, Any]]:
    paths = glob.glob(os.path.join(results_root, "**", "metrics.json"), recursive=True)
    runs = []
    for p in paths:
        try:
            with open(p, "r") as f:
                j = json.load(f)
            j["_path"] = p
            runs.append(j)
        except Exception:
            continue
    return runs


def _smooth_rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean that ignores NaNs (keeps NaN when all values in window are NaN)."""
    window = int(window)
    if window <= 1:
        return y

    out = np.full_like(y, np.nan, dtype=float)
    n = len(y)
    half = window // 2

    for i in range(n):
        j0 = max(0, i - half)
        j1 = min(n, i + half + 1)
        seg = y[j0:j1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.mean(seg))

    return out


def _median_curve(curves: List[Tuple[np.ndarray, np.ndarray]], grid: np.ndarray, *, smooth_window: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate curves to a common time grid, optionally smooth, then take median over runs."""
    if not curves:
        return grid, np.full_like(grid, np.nan, dtype=float)

    ys = []
    for t, y in curves:
        if len(t) < 2:
            continue
        # ensure increasing
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        # clamp to finite
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m]
        y = y[m]
        if len(t) < 2:
            continue
        y_i = np.interp(grid, t, y, left=np.nan, right=np.nan)
        if smooth_window and smooth_window > 1:
            y_i = _smooth_rolling_mean(y_i, smooth_window)
        ys.append(y_i)

    if not ys:
        return grid, np.full_like(grid, np.nan, dtype=float)

    Y = np.stack(ys, axis=0)
    return grid, np.nanmedian(Y, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="bench/results/raw")

    # If --domain is provided, write a single plot to --out_png (backwards compatible).
    # If --domain is omitted, write separate plots for CIFAR-10 and CHB.
    ap.add_argument("--out_png", type=str, default=None, help="Output PNG path for single-domain mode")
    ap.add_argument("--out_png_cifar", type=str, default="bench/results/aggregated/accuracy_over_time_cifar10.png")
    ap.add_argument("--out_png_chb", type=str, default="bench/results/aggregated/accuracy_over_time_chb.png")

    ap.add_argument("--domain", type=str, default=None, help="optional: cifar10 | chb. If omitted, plots both.")
    ap.add_argument("--regime", type=str, default=None, help="optional: fixed_time | fixed_compute")
    ap.add_argument("--max_minutes", type=float, default=None, help="optional x-limit")
    ap.add_argument("--grid_points", type=int, default=200)
    ap.add_argument("--smooth_window", type=int, default=0, help="Optional rolling mean window (in points). 0 disables smoothing.")
    ap.add_argument("--show_test_auc", action="store_true")
    ap.add_argument(
        "--show_variance",
        action="store_true",
        help="If set, show variance visualization (per-seed faint lines + min–max band).",
    )
    ap.add_argument(
        "--log_time",
        action="store_true",
        help="If set, use a log-scaled x-axis (wallclock time).",
    )
    ap.add_argument(
        "--chb_metric",
        type=str,
        default="val_auc",
        choices=["train_acc", "val_auc", "test_auc"],
        help="For CHB curves: which metric to plot over time (default: val_auc)",
    )
    args = ap.parse_args()

    def plot_one(domain: str, out_png: str) -> None:
        runs = load_runs(args.results_root)

        # group by optimizer
        groups = defaultdict(list)
        for r in runs:
            if r.get("domain") != domain:
                continue
            if args.regime is not None and r.get("regime") != args.regime:
                continue

            curve = r.get("curve") or {}
            t = np.asarray(curve.get("t_s") or [], dtype=float)

            if domain == "cifar10":
                y = np.asarray(curve.get("acc") or [], dtype=float)
                ylabel = "Accuracy"
            else:
                if args.chb_metric == "train_acc":
                    y = np.asarray(curve.get("train_acc") or [], dtype=float)
                    ylabel = "Train accuracy"
                elif args.chb_metric == "test_auc" or args.show_test_auc:
                    y = np.asarray(curve.get("test_auc") or [], dtype=float)
                    ylabel = "Test AUC"
                else:
                    y = np.asarray(curve.get("val_auc") or [], dtype=float)
                    ylabel = "Val AUC"

            if len(t) < 2 or len(y) < 2:
                continue

            opt = str(r.get("optimizer", "unknown"))

            # Do not show standalone SAM on CHB plots.
            if domain == "chb" and str(opt).lower() == "sam":
                continue

            groups[opt].append((t, y))

        if not groups:
            raise SystemExit(f"No curve data found for domain={domain}. Re-run training after adding curve logging.")

        # time grid based on global max
        all_tmax = [float(np.nanmax(t)) for curves in groups.values() for (t, _) in curves if len(t)]
        tmax = float(np.nanmax(all_tmax))
        if args.max_minutes is not None:
            tmax = min(tmax, float(args.max_minutes) * 60.0)

        grid = np.linspace(0.0, tmax, int(args.grid_points))

        plt.figure(figsize=(9, 5))

        if args.log_time:
            # Log scale cannot include 0.
            # We ensure the plotting grid excludes 0 by masking later (valid points).
            plt.xscale("log")
        # Print final (end-of-curve) values for each optimizer for quick debugging.
        # We define "final" as the last finite median value on the plotted time grid.
        final_rows = []  # (opt, t_final_s, y_final)

        for opt, curves in sorted(groups.items()):
            c = optimizer_color(opt)
            # Interpolate each seed curve to the common grid
            ys = []
            for t, y in curves:
                if len(t) < 2:
                    continue
                order = np.argsort(t)
                t = t[order]
                y = y[order]
                m = np.isfinite(t) & np.isfinite(y)
                t = t[m]
                y = y[m]
                if len(t) < 2:
                    continue
                y_i = np.interp(grid, t, y, left=np.nan, right=np.nan)
                if int(args.smooth_window) and int(args.smooth_window) > 1:
                    y_i = _smooth_rolling_mean(y_i, int(args.smooth_window))
                ys.append(y_i)

            if not ys:
                continue

            Y = np.stack(ys, axis=0)  # [n_seeds, n_grid]

            # Some grid locations can be NaN for *all* seeds (e.g., outside overlap of logged times).
            # Mask those columns to avoid RuntimeWarnings from nan{median,min,max}.
            valid = np.any(np.isfinite(Y), axis=0)
            if not np.any(valid):
                continue

            Yv = Y[:, valid]
            gv = grid[valid]

            # Aggregate curve (median across seeds)
            y_med = np.nanmedian(Yv, axis=0)

            # Uncertainty band (recommended for n=3 seeds): min–max envelope
            y_lo = np.nanmin(Yv, axis=0)
            y_hi = np.nanmax(Yv, axis=0)

            if args.show_variance:
                # Plot per-seed curves faintly (helps spot instability/outliers)
                for y_i in Yv:
                    plt.plot(gv, y_i, linewidth=0.9, alpha=0.20, color=c)

                # Plot band
                plt.fill_between(gv, y_lo, y_hi, alpha=0.12, color=c)

            # Plot median (main curve)
            plt.plot(gv, y_med, label=opt, linewidth=1.4, color=c)

            # Record final values (last finite point)
            mfin = np.isfinite(gv) & np.isfinite(y_med)
            if np.any(mfin):
                t_fin = float(gv[mfin][-1])
                y_fin = float(y_med[mfin][-1])
                final_rows.append((opt, t_fin, y_fin))

        if final_rows:
            print(f"\n=== Final values (domain={domain}, regime={args.regime}) ===")
            print("optimizer\twallclock_s\tmetric")
            for opt, t_fin, y_fin in sorted(final_rows, key=lambda r: r[1]):
                print(f"{opt}\t{t_fin:.6g}\t{y_fin:.6g}")

        # Make the plot start at the beginning of the x-axis.
        # For log-scale, we must start at a strictly positive value.
        if args.log_time:
            xs_pos = []
            for line in plt.gca().get_lines():
                xdata = np.asarray(line.get_xdata(), dtype=float)
                xdata = xdata[np.isfinite(xdata) & (xdata > 0)]
                if xdata.size:
                    xs_pos.append(float(np.min(xdata)))
            xmin = min(xs_pos) if xs_pos else 1e-3
            plt.xlim(left=xmin)
        else:
            plt.xlim(left=0.0)

        # Scientific notation for wallclock time
        ax = plt.gca()
        try:
            import matplotlib.ticker as mticker

            sf = mticker.ScalarFormatter(useMathText=True)
            sf.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(sf)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        except Exception:
            pass

        plt.xlabel("Wallclock time (seconds)")
        plt.ylabel(ylabel)
        # Minimalist styling
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # No title / grid
        plt.legend(fontsize=9)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.tight_layout()

        # Save PNG + matching PDF (same stem)
        plt.savefig(out_png, dpi=250)
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        plt.savefig(out_pdf)

        plt.close()

    # Single-domain mode (backwards compatible)
    if args.domain is not None:
        out_png = args.out_png
        if out_png is None:
            if args.log_time:
                out_png = "bench/results/aggregated/acc_tim_log_scale.png"
            else:
                out_png = f"bench/results/aggregated/accuracy_over_time_{args.domain}.png"
        plot_one(args.domain, out_png)
        return

    # Dual-domain mode: write separate pngs
    out_cifar = args.out_png_cifar
    out_chb = args.out_png_chb

    # If log-time is enabled and caller didn't override defaults, use explicit log-scale filenames.
    if args.log_time:
        if out_cifar == "bench/results/aggregated/accuracy_over_time_cifar10.png":
            out_cifar = "bench/results/aggregated/acc_tim_log_scale_cifar10.png"
        if out_chb == "bench/results/aggregated/accuracy_over_time_chb.png":
            out_chb = "bench/results/aggregated/acc_tim_log_scale_chb.png"

    plot_one("cifar10", out_cifar)
    plot_one("chb", out_chb)


if __name__ == "__main__":
    main()