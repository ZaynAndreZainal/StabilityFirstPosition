from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Paths (consistent with this repo layout)
# -------------------------

SUMMARY_JSON = "bench/results/aggregated/summary.json"
OUT_PNG = "bench/results/aggregated/parento_figures/pareto_tradeoff.png"

DOMAINS = ["cifar10", "chb"]  # Panel A, Panel B


# -------------------------
# Visual encodings
# -------------------------

# Marker by regime
REGIME_MARKERS = {
    "fixed_time": "o",
    "fixed_compute": "^",
}

# Optimizer colors (direct labeling; avoids ambiguity when families share colors)
# Use a colorblind-friendly palette (Matplotlib tab10-ish) + a neutral fallback.
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
    key = str(opt_name).lower()
    return OPT_COLORS.get(key, "#7f7f7f")

# Readability guard (Nature-style small multiples)
MAX_POINTS_PER_PANEL = 10
TOPK_PER_REGIME_IF_TOO_MANY = 5


# -------------------------
# Helpers
# -------------------------

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


# Family mapping removed.
# We color points directly by optimizer using optimizer_color(...).
# Keep this stub only for backwards compatibility in older code paths.
def optimizer_family(opt_name: str) -> str:
    return "(unused)"


def alpha_from_failure_rate(fr: Optional[float]) -> float:
    """Higher failure_rate => more transparent."""
    if fr is None or not np.isfinite(fr):
        return 0.9
    fr = float(np.clip(fr, 0.0, 1.0))
    # Map [0, 1] -> [0.90, 0.25]
    return float(0.9 - 0.65 * fr)


@dataclass
class Point:
    domain: str
    regime: str
    optimizer: str
    x_cost_s: float
    y_perf: float
    failure_rate: Optional[float]
    raw: Dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.optimizer}\n({self.regime})"


def extract_points(rows: List[Dict[str, Any]], domain: str, *, regime_filter: Optional[str] = None) -> Tuple[List[Point], List[str]]:
    pts: List[Point] = []
    warnings: List[str] = []

    for r in rows:
        if r.get("domain") != domain:
            continue

        regime = str(r.get("regime", "unknown"))
        if regime_filter is not None and regime != regime_filter:
            continue

        optimizer = str(r.get("optimizer", "unknown"))

        # Do not show standalone SAM on CHB plots.
        if domain == "chb" and str(optimizer).lower() == "sam":
            continue

        # y-axis: primary IID performance aggregated from metrics.json
        # CIFAR-10: test accuracy; CHB: test AUC (from updated train_chb.py)
        y = _safe_float(r.get("iid_mean"))
        if y is None:
            warnings.append(f"[{domain}] excluded {optimizer} ({regime}): missing/invalid iid_mean")
            continue

        # Prefer time_to_target_s_mean; fallback to aggregated wallclock budget.
        x = _safe_float(r.get("time_to_target_s_mean"))
        if x is None:
            x = _safe_float(r.get("budget_wallclock_seconds_mean"))

        if x is None:
            warnings.append(
                f"[{domain}] excluded {optimizer} ({regime}): missing time_to_target_s_mean and budget_wallclock_seconds_mean"
            )
            continue

        fr = _safe_float(r.get("failure_rate"))
        pts.append(Point(domain=domain, regime=regime, optimizer=optimizer, x_cost_s=x, y_perf=y, failure_rate=fr, raw=r))

    return pts, warnings


def pareto_front(points: List[Point]) -> List[Point]:
    """Pareto-optimal for minimizing x and maximizing y."""
    if not points:
        return []

    ps = sorted(points, key=lambda p: (p.x_cost_s, -p.y_perf))

    frontier: List[Point] = []
    best_y = -float("inf")
    for p in ps:
        if p.y_perf > best_y + 1e-12:
            frontier.append(p)
            best_y = p.y_perf

    return frontier


def step_polyline(frontier: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    if not frontier:
        return np.array([]), np.array([])

    fs = sorted(frontier, key=lambda p: p.x_cost_s)

    xs: List[float] = [fs[0].x_cost_s]
    ys: List[float] = [fs[0].y_perf]

    for p in fs[1:]:
        # horizontal move
        xs.append(p.x_cost_s)
        ys.append(ys[-1])
        # vertical move
        xs.append(p.x_cost_s)
        ys.append(p.y_perf)

    return np.array(xs), np.array(ys)


def choose_labels(points: List[Point], frontier: List[Point]) -> List[Point]:
    """Label Pareto points + extreme outliers (top perf, worst efficiency)."""
    if not points:
        return []

    selected: List[Point] = []
    seen = set()

    def add(p: Point):
        if id(p) in seen:
            return
        selected.append(p)
        seen.add(id(p))

    for p in frontier:
        add(p)

    for p in sorted(points, key=lambda p: p.y_perf, reverse=True)[:2]:
        add(p)

    for p in sorted(points, key=lambda p: p.x_cost_s, reverse=True)[:2]:
        add(p)

    return selected


def maybe_downselect(points: List[Point]) -> List[Point]:
    if len(points) <= MAX_POINTS_PER_PANEL:
        return points

    out: List[Point] = []
    regimes = sorted(set(p.regime for p in points))
    for rg in regimes:
        ps = [p for p in points if p.regime == rg]
        ps = sorted(ps, key=lambda p: p.y_perf, reverse=True)[:TOPK_PER_REGIME_IF_TOO_MANY]
        out.extend(ps)

    if len(out) > MAX_POINTS_PER_PANEL:
        out = sorted(out, key=lambda p: p.y_perf, reverse=True)[:MAX_POINTS_PER_PANEL]

    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", type=str, default=SUMMARY_JSON)
    ap.add_argument("--out_png", type=str, default=OUT_PNG)
    ap.add_argument(
        "--out_pdf",
        type=str,
        default=None,
        help="Optional. If omitted, we also save a PDF next to --out_png (same stem).",
    )

    ap.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["cifar10", "chb"],
        help="Optional: plot only one domain (cifar10 or chb)",
    )

    ap.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=["fixed_time", "fixed_compute"],
        help="Optional: plot only one regime",
    )

    ap.add_argument(
        "--no_downsampling",
        action="store_true",
        help="If set, plot all points (disable readability downsampling).",
    )

    args = ap.parse_args()

    # Track whether the caller explicitly provided --out_png.
    # We only auto-rename outputs when they kept the default.
    _out_png_arg = args.out_png

    # If the user didn't override the output path, save to a more specific filename
    # that reflects filters (domain/regime) for convenience.
    if args.out_png == OUT_PNG and (args.domain is not None or args.regime is not None):
        parts = ["pareto"]
        if args.domain is not None:
            parts.append(args.domain)
        if args.regime is not None:
            parts.append(args.regime)
        args.out_png = os.path.join(os.path.dirname(OUT_PNG), "_".join(parts) + ".png")

    # NOTE: We only append a "downsampled" suffix if downselection actually happens.
    # We determine that after extracting points (later), by comparing counts.

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    rows = summary.get("groups", [])
    if not isinstance(rows, list):
        raise ValueError("summary['groups'] must be a list")

    all_warnings: List[str] = []
    pareto_report: Dict[str, List[str]] = {}

    plot_domains = DOMAINS if args.domain is None else [args.domain]

    # Layout: if a single domain is requested, use a square figure.
    # Otherwise, use a wide 2-panel layout.
    #
    # IMPORTANT: we intentionally do *not* use constrained_layout here.
    # We want a predictable bottom margin so any notes can live *below* the x-axis
    # (outside the plotting area) without overlapping points.
    if len(plot_domains) == 1:
        fig, ax0 = plt.subplots(1, 1, figsize=(5.2, 5.2))
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, len(plot_domains), figsize=(10.5, 4.2))

    # Reserve extra space under axes for:
    # - the downselection note
    # - a figure-level legend placed below the x-axis
    fig.subplots_adjust(bottom=0.32)

    # Note: for single-domain (square) output, we keep all text inside the canvas
    # so the *saved image* remains square (avoid bbox expansion).
    # if len(plot_domains) == 1:
    #     fig.text(
    #         0.5,
    #         0.01,
    #         "Error bars omitted (per-seed not provided). Transparency indicates failure rate (more transparent = higher failure). Pareto-optimal points have a thicker outline.",
    #         ha="center",
    #         va="bottom",
    #         fontsize=9,
    #     )
    # else:
    #     fig.text(
    #         0.5,
    #         -0.02,
    #         "Error bars omitted (per-seed not provided). Transparency indicates failure rate (more transparent = higher failure). Pareto-optimal points have a thicker outline.",
    #         ha="center",
    #         va="top",
    #         fontsize=9,
    #     )

    panel_titles = {
        "cifar10": "CIFAR-10 (Test accuracy)",
        "chb": "CHB-MIT (Test AUC; time-to-target uses Val AUC)",
    }

    any_downselected = False

    for ax, domain in zip(axes, plot_domains):
        pts, warns = extract_points(rows, domain=domain, regime_filter=args.regime)
        all_warnings.extend(warns)

        # Full frontier for reporting
        frontier_full = pareto_front(pts)
        pareto_report[domain] = [f"{p.optimizer} ({p.regime})" for p in frontier_full]

        # Downselect only for plot readability (unless disabled)
        if args.no_downsampling:
            pts_plot = pts
        else:
            pts_plot = maybe_downselect(pts)
            if len(pts_plot) < len(pts):
                any_downselected = True

        frontier = pareto_front(pts_plot)

        # Axis / style (minimal)
        # If plotting both domains, keep explicit panel labels. If plotting a single domain,
        # keep the title minimal.
        # if args.domain is None:
        #     prefix = "Panel A: " if domain == "cifar10" else "Panel B: "
        #     ax.set_title(prefix + panel_titles.get(domain, domain), fontsize=11)
        # else:
        #     ax.set_title(panel_titles.get(domain, domain), fontsize=11)
        ax.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(False)

        ax.set_xlabel("Time-to-target (s)", fontsize=10)

        # Scientific notation for x-axis (seconds)
        try:
            import matplotlib.ticker as mticker

            sf = mticker.ScalarFormatter(useMathText=True)
            sf.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(sf)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        except Exception:
            pass

        # Only show the y-axis label on the left panel when plotting multiple domains.
        # This reduces clutter in 2-panel figures.
        if len(plot_domains) > 1 and domain != plot_domains[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Performance", fontsize=10)

        # Points
        # Color encodes optimizer identity (see OPT_COLORS).
        # Marker shape encodes budget regime.
        for p in pts_plot:
            color = optimizer_color(p.optimizer)
            marker = REGIME_MARKERS.get(p.regime, "s")
            a = alpha_from_failure_rate(p.failure_rate)

            ax.scatter(
                p.x_cost_s,
                p.y_perf,
                s=70,
                marker=marker,
                alpha=min(1.0, a + 0.10),
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )

        # Emphasize Pareto points (keep filled markers; no hollow encoding)
        for p in frontier:
            ax.scatter(
                p.x_cost_s,
                p.y_perf,
                s=110,
                marker=REGIME_MARKERS.get(p.regime, "s"),
                alpha=1.0,
                color=optimizer_color(p.optimizer),
                edgecolors="black",
                linewidths=2.0,
                zorder=4,
            )

        # Frontier polyline
        fx, fy = step_polyline(frontier)
        if fx.size:
            ax.plot(fx, fy, color="black", linewidth=1.2, zorder=2)

        # No per-point labels (keeps figure uncluttered)

        # Legend (optimizer colors + regime markers)
        from matplotlib.lines import Line2D

        opts_in_panel = sorted({str(p.optimizer).lower() for p in pts_plot})
        regimes_in_panel = sorted({p.regime for p in pts_plot})

        handles = []
        labels = []

        for opt in opts_in_panel:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=optimizer_color(opt),
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    markersize=7,
                )
            )
            labels.append(opt)

        # Only include regime legend entries when we are *not* filtering to a single regime.
        # If --regime is specified, the marker shape no longer encodes additional information.
        if args.regime is None:
            for rg in regimes_in_panel:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=REGIME_MARKERS.get(rg, "s"),
                        linestyle="None",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markersize=7,
                    )
                )
                labels.append(rg)

        # Defer legend rendering: we draw a single figure-level legend below the x-axis.
        # Keep one set of handles/labels (they are identical across panels under the same filters).
        if "_fig_legend" not in locals():
            _fig_legend = (handles, labels)

        if (not args.no_downsampling) and (len(pts) > len(pts_plot)):
            # Place the note *below the x-axis* (outside the axes).
            # Using an axes-anchored annotation avoids it drifting into the plot when
            # layout/bbox adjustments change figure coordinates.
            ax.annotate(
                f"Showing {len(pts_plot)}/{len(pts)} points (downselected for readability)",
                xy=(0.0, 0.0),
                xycoords="axes fraction",
                xytext=(0, -28),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8.5,
                annotation_clip=False,
            )

    # Figure-level legend (below x-axis)
    if "_fig_legend" in locals():
        handles, labels = _fig_legend
        # Put the legend below the axes, centered. Use multiple columns to keep it compact.
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=max(1, min(4, len(labels))),
            frameon=False,
            fontsize=9,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    # If downsampling actually occurred anywhere, and the caller kept the default
    # output path, annotate the filename to make it explicit.
    if any_downselected and _out_png_arg == OUT_PNG:
        base, ext = os.path.splitext(args.out_png)
        if not base.endswith("_downsampled"):
            args.out_png = base + "_downsampled" + ext

    # Default PDF path: match the final PNG output name (same stem).
    # This must happen *after* any auto-renaming of args.out_png.
    if args.out_pdf is None:
        stem, _ext = os.path.splitext(args.out_png)
        args.out_pdf = stem + ".pdf"

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    # IMPORTANT: keep the single-domain output *strictly square*.
    # - bbox_inches="tight" can change the final pixel dimensions.
    # - the figure footnote is placed inside the canvas above.
    if len(plot_domains) == 1:
        plt.savefig(args.out_png, dpi=300)
        plt.savefig(args.out_pdf)
    else:
        plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
        plt.savefig(args.out_pdf, bbox_inches="tight")

    plt.close(fig)

        # Text summary
    print("\n=== Pareto-optimal methods (min x, max y) ===")
    for domain in DOMAINS:
        print(f"\n[{domain}]")
        ms = pareto_report.get(domain, [])
        if not ms:
            print("  (none)")
        else:
            for m in ms:
                print(f"  - {m}")

    print("\n=== Exclusions / warnings (missing efficiency or performance) ===")
    if not all_warnings:
        print("  (none)")
    else:
        for w in all_warnings:
            print(f"  - {w}")

    print(f"\nSaved: {args.out_png}")


if __name__ == "__main__":
    main()