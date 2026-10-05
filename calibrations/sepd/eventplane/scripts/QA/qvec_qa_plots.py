#!/usr/bin/env python3
"""
qvec_qa_plots.py

Publication-grade plotting library for sEPD Q-Vector run-by-run flatness QA.
Utilizes matplotlib and mplhep (ATLAS style) to produce:
1. 2D Heatmaps (Run Index vs 2Psi2)
2. 4-Panel Trend Plots vs Run Number (RMS, chi2/ndf, Harmonic Modulations, Events)
3. Metric Distribution Histograms (Unzoomed & Zoomed with statistics boxes)
4. Ensemble Average Flatness Profiles
5. 3-Arm Diagnostic Figures (NS, N, S) with Fourier Fit Curves
6. Failure Mode Galleries
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import numpy as np


def _apply_hep_style():
    """Apply ATLAS-like clean HEP styling."""
    try:
        hep.style.use("ATLAS")
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def _get_subdet_metric(d: Dict[str, Any], subdet: str) -> Dict[str, Any]:
    """Helper to get metrics for a specific subdetector from a run record."""
    if "subdetectors" in d and subdet in d["subdetectors"]:
        return d["subdetectors"][subdet]
    # Fallback to top-level if subdet is NS or already flattened
    return d


# ==============================================================================
# 1. 2D RATIO HEATMAP
# ==============================================================================

def plot_heatmap(
    data_list: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    subdet: str = "NS",
    test_runs: Optional[Sequence[int]] = None,
    vmin: float = 0.90,
    vmax: float = 1.10,
) -> None:
    """
    Generate 2D color mesh of ratio to average (y / y_mean) vs Run Index and 2Psi2.
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_data = [d for d in data_list if _get_subdet_metric(d, subdet).get("total_events", 0) > 0]
    if not valid_data:
        print(f"Warning: No valid data to plot heatmap for {subdet}")
        return

    runs = [d["run_number"] for d in valid_data]
    n_runs = len(runs)

    sample_m = _get_subdet_metric(valid_data[0], subdet)
    edges_y = sample_m["edges"]

    # Build ratio matrix: shape (n_runs, n_psi_bins)
    ratio_matrix = np.array([_get_subdet_metric(d, subdet)["ratios"] for d in valid_data])

    fig, ax = plt.subplots(figsize=(13, 6.5))

    x_grid, y_grid = np.meshgrid(np.arange(n_runs + 1), edges_y)
    c = ax.pcolormesh(
        x_grid,
        y_grid,
        ratio_matrix.T,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    cbar = fig.colorbar(c, ax=ax, pad=0.015, aspect=25, fraction=0.046)
    cbar.set_label(rf"Ratio to Mean Bin ($\mathrm{{d}}N / \mathrm{{d}}(2\Psi_2) / \langle\mathrm{{d}}N\rangle$)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    subdet_label = "North+South Combined" if subdet == "NS" else ("North Arm" if subdet == "N" else "South Arm")
    ax.set_ylabel(rf"$2\Psi_2^{{\mathrm{{{subdet}}}}}$ [rad]", fontsize=15)
    ax.set_xlabel("Run Index", fontsize=15)
    ax.set_xlim(0, n_runs)
    ax.set_ylim(-np.pi, np.pi)

    # Dynamic x-axis tick step
    if n_runs >= 2000: step = 500
    elif n_runs >= 800: step = 200
    elif n_runs >= 400: step = 100
    elif n_runs >= 100: step = 25
    elif n_runs >= 20: step = 10
    else: step = max(1, n_runs // 5)

    tick_positions = np.arange(0, n_runs + 1, step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(p) for p in tick_positions], fontsize=12)

    # Highlight specific test runs
    if test_runs:
        for tr in test_runs:
            if tr in runs:
                idx = runs.index(tr)
                ax.axvline(idx + 0.5, color="black", linestyle="--", linewidth=1.8, alpha=0.9)
                ha = "left" if idx < n_runs * 0.06 else ("right" if idx > n_runs * 0.94 else "center")
                ax.text(
                    idx + 0.5,
                    np.pi + 0.15,
                    f"Run {tr}",
                    rotation=0,
                    ha=ha,
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="darkred",
                    clip_on=False,
                )

    ax.text(
        0.02,
        0.94,
        f"sEPD Q-Vector Flattening QA: {subdet_label}\nRuns: {n_runs:,}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 2. 4-PANEL TREND PLOTS VS RUN NUMBER
# ==============================================================================

def _draw_rms_scatter(ax, runs, rms_vals, is_good, is_outlier, is_test, max_rms_pct, s_good=28, s_bad=42):
    ax.scatter(runs[is_good], rms_vals[is_good], color="forestgreen", s=s_good, alpha=0.7, label="Good Runs", edgecolors="none")
    ax.scatter(runs[is_outlier], rms_vals[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^", label="Flagged Runs")
    if np.any(is_test):
        ax.scatter(runs[is_test], rms_vals[is_test], facecolors="none", edgecolors="black", s=130, linewidths=2.0, marker="o", label="Test Runs")
        for r, val in zip(runs[is_test], rms_vals[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(max_rms_pct, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_rms_pct:.1f}%)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.35)


def _draw_chi2_scatter(ax, runs, chi2_vals, is_good, is_outlier, is_test, max_chi2, s_good=28, s_bad=42):
    ax.scatter(runs[is_good], chi2_vals[is_good], color="forestgreen", s=s_good, alpha=0.7, edgecolors="none")
    ax.scatter(runs[is_outlier], chi2_vals[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^")
    if np.any(is_test):
        ax.scatter(runs[is_test], chi2_vals[is_test], facecolors="none", edgecolors="black", s=130, linewidths=2.0, marker="o")
        for r, val in zip(runs[is_test], chi2_vals[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.3, label="Nominal (1.0)")
    ax.axhline(max_chi2, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_chi2:.1f})")
    ax.set_ylim(bottom=0.0, top=max(3.5, np.nanpercentile(chi2_vals, 99.5) * 1.2) if len(chi2_vals) else 3.5)
    ax.grid(True, linestyle="--", alpha=0.35)


def _draw_modulations_scatter(ax, runs, a1_vals, a2_vals, max_a, s=28):
    ax.scatter(runs, a1_vals, color="royalblue", s=s, alpha=0.6, marker="o", label=r"$A_1$ (Dipole Residual)")
    ax.scatter(runs, a2_vals, color="mediumvioletred", s=s, alpha=0.6, marker="s", label=r"$A_2$ (Quadrupole Residual)")
    ax.axhline(max_a, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_a:.1f}%)")
    ax.set_ylim(bottom=0.0, top=max(1.5, max_a * 1.5))
    ax.grid(True, linestyle="--", alpha=0.35)


def _draw_events_scatter(ax, runs, nevts, is_good, is_outlier, is_test, min_events, s_good=28, s_bad=42, label_runs=False):
    ax.scatter(runs[is_good], nevts[is_good], color="forestgreen", s=s_good, alpha=0.7, edgecolors="none", label="Good Runs" if label_runs else None)
    ax.scatter(runs[is_outlier], nevts[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^", label="Flagged Runs" if label_runs else None)
    if np.any(is_test):
        ax.scatter(runs[is_test], nevts[is_test], facecolors="none", edgecolors="black", s=130, linewidths=2.0, marker="o", label="Test Runs" if label_runs else None)
        for r, val in zip(runs[is_test], nevts[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(min_events, color="crimson", linestyle="--", linewidth=1.8, label=rf"Min Evts ({min_events:,.0f})")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_trends(
    data_list: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    subdet: str = "NS",
    test_runs: Optional[Sequence[int]] = None,
    max_rms_pct: float = 3.5,
    max_chi2: float = 2.0,
    max_a: float = 1.0,
    min_events: float = 100000.0,
    save_individual: bool = True,
) -> None:
    """
    Generate 4-panel trends vs Run Number (RMS, chi2/ndf, Fourier Modulations, Event Counts).
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = np.array([d["run_number"] for d in data_list])
    m_list = [_get_subdet_metric(d, subdet) for d in data_list]

    rms_vals = np.array([m.get("rms_pct", np.nan) for m in m_list])
    chi2_vals = np.array([m.get("chi2_ndf", np.nan) for m in m_list])
    a1_vals = np.array([m.get("A1_pct", np.nan) for m in m_list])
    a2_vals = np.array([m.get("A2_pct", np.nan) for m in m_list])
    nevts = np.array([m.get("total_events", 0.0) for m in m_list])

    statuses = [m.get("status", "GOOD") for m in m_list]
    is_good = np.array([s == "GOOD" for s in statuses])
    is_outlier = ~is_good

    test_run_set = set(test_runs or [])
    is_test = np.array([r in test_run_set for r in runs])

    subdet_label = "North+South Combined" if subdet == "NS" else ("North Arm" if subdet == "N" else "South Arm")

    # 4-panel canvas
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)

    _draw_rms_scatter(axes[0], runs, rms_vals, is_good, is_outlier, is_test, max_rms_pct)
    axes[0].set_ylabel(rf"{subdet} RMS Non-Flatness [%]", fontsize=13)
    axes[0].legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)

    _draw_chi2_scatter(axes[1], runs, chi2_vals, is_good, is_outlier, is_test, max_chi2)
    axes[1].set_ylabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat", fontsize=13)
    axes[1].legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)

    _draw_modulations_scatter(axes[2], runs, a1_vals, a2_vals, max_a)
    axes[2].set_ylabel(rf"{subdet} Modulation $A_k$ [%]", fontsize=13)
    axes[2].legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)

    _draw_events_scatter(axes[3], runs, nevts, is_good, is_outlier, is_test, min_events)
    axes[3].set_ylabel("Total Events", fontsize=13)
    axes[3].set_xlabel("Run Number", loc="center", fontsize=15)
    axes[3].legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save individual panels if requested
    if save_individual:
        stem = output_path.stem
        parent = output_path.parent

        # 1. RMS
        fig_r, ax_r = plt.subplots(figsize=(11, 6.5))
        _draw_rms_scatter(ax_r, runs, rms_vals, is_good, is_outlier, is_test, max_rms_pct, s_good=32, s_bad=48)
        ax_r.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_r.set_ylabel(rf"{subdet} RMS Non-Flatness [%]", fontsize=14)
        ax_r.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_r.tight_layout()
        fig_r.savefig(parent / f"{stem}_rms.png", dpi=300, bbox_inches="tight")
        plt.close(fig_r)

        # 2. Chi2
        fig_c, ax_c = plt.subplots(figsize=(11, 6.5))
        _draw_chi2_scatter(ax_c, runs, chi2_vals, is_good, is_outlier, is_test, max_chi2, s_good=32, s_bad=48)
        ax_c.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_c.set_ylabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat", fontsize=14)
        ax_c.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_c.tight_layout()
        fig_c.savefig(parent / f"{stem}_chi2.png", dpi=300, bbox_inches="tight")
        plt.close(fig_c)

        # 3. Modulation
        fig_m, ax_m = plt.subplots(figsize=(11, 6.5))
        _draw_modulations_scatter(ax_m, runs, a1_vals, a2_vals, max_a, s=32)
        ax_m.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_m.set_ylabel(rf"{subdet} Harmonic Modulation Amplitude [%]", fontsize=14)
        ax_m.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_m.tight_layout()
        fig_m.savefig(parent / f"{stem}_modulation.png", dpi=300, bbox_inches="tight")
        plt.close(fig_m)

        # 4. Events
        fig_e, ax_e = plt.subplots(figsize=(11, 6.5))
        _draw_events_scatter(ax_e, runs, nevts, is_good, is_outlier, is_test, min_events, s_good=32, s_bad=48, label_runs=True)
        ax_e.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_e.set_ylabel("Total Events", fontsize=14)
        ax_e.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)
        fig_e.tight_layout()
        fig_e.savefig(parent / f"{stem}_events.png", dpi=300, bbox_inches="tight")
        plt.close(fig_e)


# ==============================================================================
# 3. METRIC DISTRIBUTION HISTOGRAMS
# ==============================================================================

def _draw_rms_hist(ax, rms_vals, max_rms_pct, zoomed=False, title_font=9.5):
    if len(rms_vals) > 0:
        n_pass = sum(1 for v in rms_vals if v <= max_rms_pct)
        n_fail = len(rms_vals) - n_pass
        pct_pass = 100.0 * n_pass / len(rms_vals)
        pct_fail = 100.0 * n_fail / len(rms_vals)

        if zoomed:
            x_max = 2.0
            data = np.clip(rms_vals, 0, x_max)
            bins = np.linspace(0, x_max, 40)
            extra_txt = f"\n(Clipped at {x_max:.1f}%)"
        else:
            data = rms_vals
            x_max = max(max_rms_pct * 1.05, max(rms_vals) * 1.05) if len(rms_vals) else max_rms_pct * 1.1
            bins = np.linspace(0, x_max, 45)
            extra_txt = f"\nMax RMS: {max(rms_vals):.2f}%"

        ax.hist(data, bins=bins, histtype="step", color="steelblue", linewidth=2.0, label="Runs")
        if max_rms_pct <= x_max:
            ax.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct:.1f}%)")
        ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.97), fontsize=title_font, frameon=True, framealpha=0.92)

        box_text = (
            rf"Pass ($\leq {max_rms_pct}\%$): {n_pass:,} ({pct_pass:.1f}%)" + "\n" +
            rf"Fail (> {max_rms_pct}%): {n_fail:,} ({pct_fail:.1f}%)" + extra_txt
        )
        ax.text(0.97, 0.76, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92))
        ax.set_xlim(0, x_max)

    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35, which="both")


def _draw_chi2_hist(ax, chi2_vals, max_chi2, zoomed=False, title_font=9.5):
    if len(chi2_vals) > 0:
        n_pass = sum(1 for v in chi2_vals if v <= max_chi2)
        n_fail = len(chi2_vals) - n_pass
        pct_pass = 100.0 * n_pass / len(chi2_vals)
        pct_fail = 100.0 * n_fail / len(chi2_vals)

        if zoomed:
            x_max = 1.4
            data = np.clip(chi2_vals, 0, x_max)
            bins = np.linspace(0, x_max, 40)
            extra_txt = f"\n(Clipped at {x_max:.1f})"
        else:
            data = chi2_vals
            x_max = max(max_chi2 * 1.05, max(chi2_vals) * 1.05) if len(chi2_vals) else max_chi2 * 1.1
            bins = np.linspace(0, x_max, 45)
            extra_txt = "\n" + rf"Max $\chi^2/\mathrm{{ndf}}$: {max(chi2_vals):.2f}"

        ax.hist(data, bins=bins, histtype="step", color="forestgreen", linewidth=2.0, label="Runs")
        if max_chi2 <= x_max:
            ax.axvline(max_chi2, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_chi2:.1f})")
        ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.97), fontsize=title_font, frameon=True, framealpha=0.92)

        box_text = (
            rf"Pass ($\leq {max_chi2:.1f}$): {n_pass:,} ({pct_pass:.1f}%)" + "\n" +
            rf"Fail (> {max_chi2:.1f}): {n_fail:,} ({pct_fail:.1f}%)" + extra_txt
        )
        ax.text(0.97, 0.76, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92))
        ax.set_xlim(0, x_max)

    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35, which="both")


def _draw_modulations_hist(ax, a1_vals, a2_vals, max_a, zoomed=False, title_font=9.5):
    if len(a1_vals) > 0 and len(a2_vals) > 0:
        if zoomed:
            x_max = 0.8
            d1 = np.clip(a1_vals, 0, x_max)
            d2 = np.clip(a2_vals, 0, x_max)
            bins = np.linspace(0, x_max, 40)
            extra_txt = f"\n(Clipped at {x_max:.1f}%)"
        else:
            d1 = a1_vals
            d2 = a2_vals
            max_val = max(max(a1_vals), max(a2_vals))
            x_max = max(max_a * 1.08, max_val * 1.08)
            bins = np.linspace(0, x_max, 45)
            extra_txt = f"\nMax $A_1$: {max(a1_vals):.2f}%, $A_2$: {max(a2_vals):.2f}%"

        ax.hist(d1, bins=bins, histtype="step", color="royalblue", linewidth=2.0, label=r"$A_1$ (Dipole)")
        ax.hist(d2, bins=bins, histtype="step", color="mediumvioletred", linewidth=2.0, label=r"$A_2$ (Quadrupole)")
        if max_a <= x_max:
            ax.axvline(max_a, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_a:.1f}%)")
        ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.97), fontsize=title_font, frameon=True, framealpha=0.92)

        n_fail = sum(1 for v1, v2 in zip(a1_vals, a2_vals) if v1 > max_a or v2 > max_a)
        pct_fail = 100.0 * n_fail / len(a1_vals)

        box_text = (
            rf"Fail ($A_1$ or $A_2 > {max_a}\%$): {n_fail:,} ({pct_fail:.1f}%)" + extra_txt
        )
        ax.text(0.97, 0.68, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92))
        ax.set_xlim(0, x_max)

    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35, which="both")


def _draw_events_hist(ax, nevts, min_events, title_font=9.5):
    if len(nevts) > 0:
        x_min = max(1.0, min(nevts) * 0.85)
        x_max = max(nevts) * 1.15
        log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 40)
        ax.hist(nevts, bins=log_bins, histtype="step", color="coral", linewidth=2.0, label="Runs")
        ax.axvline(min_events, color="crimson", linestyle="--", linewidth=2, label=rf"Min ({min_events:,.0f})")
        ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.97), fontsize=title_font, frameon=True, framealpha=0.92)

        n_fail = sum(1 for v in nevts if v < min_events)
        pct_fail = 100.0 * n_fail / len(nevts)

        box_text = (
            rf"Total Runs: {len(nevts):,}" + "\n" +
            rf"Median: {float(np.median(nevts)):.2e} evts" + "\n" +
            rf"Low Evts (< {min_events:,.0f}): {n_fail} ({pct_fail:.1f}%)"
        )
        ax.text(0.04, 0.76, box_text, transform=ax.transAxes, ha="left", va="top", fontsize=title_font,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92))
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)

    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35, which="both")


def plot_metric_distributions(
    data_list: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    subdet: str = "NS",
    max_rms_pct: float = 3.5,
    max_chi2: float = 2.0,
    max_a: float = 1.0,
    min_events: float = 100000.0,
    save_individual: bool = True,
) -> None:
    """
    Generate 4-panel histogram distributions of scalar QA metrics (unzoomed and zoomed),
    and optionally save individual single-metric distribution plots.
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    m_list = [_get_subdet_metric(d, subdet) for d in data_list]
    rms_vals = [m["rms_pct"] for m in m_list if not np.isnan(m.get("rms_pct", np.nan))]
    chi2_vals = [m["chi2_ndf"] for m in m_list if not np.isnan(m.get("chi2_ndf", np.nan))]
    a1_vals = [m["A1_pct"] for m in m_list if not np.isnan(m.get("A1_pct", np.nan))]
    a2_vals = [m["A2_pct"] for m in m_list if not np.isnan(m.get("A2_pct", np.nan))]
    nevts = [m["total_events"] for m in m_list if m.get("total_events", 0) > 0]

    # 1. Unzoomed 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    _draw_rms_hist(axes[0, 0], rms_vals, max_rms_pct, zoomed=False)
    axes[0, 0].set_xlabel(rf"{subdet} RMS Non-Flatness [%]")
    axes[0, 0].set_ylabel("Runs")

    _draw_chi2_hist(axes[0, 1], chi2_vals, max_chi2, zoomed=False)
    axes[0, 1].set_xlabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat")
    axes[0, 1].set_ylabel("Runs")

    _draw_modulations_hist(axes[1, 0], a1_vals, a2_vals, max_a, zoomed=False)
    axes[1, 0].set_xlabel(rf"{subdet} Harmonic Modulation Amplitude [%]")
    axes[1, 0].set_ylabel("Runs")

    _draw_events_hist(axes[1, 1], nevts, min_events)
    axes[1, 1].set_xlabel("Total Events per Run")
    axes[1, 1].set_ylabel("Runs")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Zoomed 4-panel
    zoomed_path = output_path.parent / f"{output_path.stem}_zoomed.png"
    fig_z, axes_z = plt.subplots(2, 2, figsize=(12, 10))

    _draw_rms_hist(axes_z[0, 0], rms_vals, max_rms_pct, zoomed=True)
    axes_z[0, 0].set_xlabel(rf"{subdet} RMS Non-Flatness [%]")
    axes_z[0, 0].set_ylabel("Runs")

    _draw_chi2_hist(axes_z[0, 1], chi2_vals, max_chi2, zoomed=True)
    axes_z[0, 1].set_xlabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat")
    axes_z[0, 1].set_ylabel("Runs")

    _draw_modulations_hist(axes_z[1, 0], a1_vals, a2_vals, max_a, zoomed=True)
    axes_z[1, 0].set_xlabel(rf"{subdet} Harmonic Modulation Amplitude [%]")
    axes_z[1, 0].set_ylabel("Runs")

    _draw_events_hist(axes_z[1, 1], nevts, min_events)
    axes_z[1, 1].set_xlabel("Total Events per Run")
    axes_z[1, 1].set_ylabel("Runs")

    fig_z.tight_layout()
    fig_z.savefig(zoomed_path, dpi=300, bbox_inches="tight")
    plt.close(fig_z)

    # 3. Individual Single-Metric Distribution Plots
    if save_individual:
        parent = output_path.parent

        # RMS
        fig_r, ax_r = plt.subplots(figsize=(8.5, 6.5))
        _draw_rms_hist(ax_r, rms_vals, max_rms_pct, zoomed=False, title_font=10.0)
        ax_r.set_xlabel(rf"{subdet} RMS Non-Flatness [%]", fontsize=14)
        ax_r.set_ylabel("Runs", fontsize=14)
        fig_r.tight_layout()
        fig_r.savefig(parent / f"rms_distribution_{subdet}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_r)

        fig_rz, ax_rz = plt.subplots(figsize=(8.5, 6.5))
        _draw_rms_hist(ax_rz, rms_vals, max_rms_pct, zoomed=True, title_font=10.0)
        ax_rz.set_xlabel(rf"{subdet} RMS Non-Flatness [%]", fontsize=14)
        ax_rz.set_ylabel("Runs", fontsize=14)
        fig_rz.tight_layout()
        fig_rz.savefig(parent / f"rms_distribution_{subdet}_zoomed.png", dpi=300, bbox_inches="tight")
        plt.close(fig_rz)

        # Chi2 / ndf
        fig_c, ax_c = plt.subplots(figsize=(8.5, 6.5))
        _draw_chi2_hist(ax_c, chi2_vals, max_chi2, zoomed=False, title_font=10.0)
        ax_c.set_xlabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat", fontsize=14)
        ax_c.set_ylabel("Runs", fontsize=14)
        fig_c.tight_layout()
        fig_c.savefig(parent / f"chi2_distribution_{subdet}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_c)

        fig_cz, ax_cz = plt.subplots(figsize=(8.5, 6.5))
        _draw_chi2_hist(ax_cz, chi2_vals, max_chi2, zoomed=True, title_font=10.0)
        ax_cz.set_xlabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat", fontsize=14)
        ax_cz.set_ylabel("Runs", fontsize=14)
        fig_cz.tight_layout()
        fig_cz.savefig(parent / f"chi2_distribution_{subdet}_zoomed.png", dpi=300, bbox_inches="tight")
        plt.close(fig_cz)

        # Modulations A1, A2
        fig_m, ax_m = plt.subplots(figsize=(8.5, 6.5))
        _draw_modulations_hist(ax_m, a1_vals, a2_vals, max_a, zoomed=False, title_font=10.0)
        ax_m.set_xlabel(rf"{subdet} Harmonic Modulation Amplitude [%]", fontsize=14)
        ax_m.set_ylabel("Runs", fontsize=14)
        fig_m.tight_layout()
        fig_m.savefig(parent / f"modulation_distribution_{subdet}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_m)

        fig_mz, ax_mz = plt.subplots(figsize=(8.5, 6.5))
        _draw_modulations_hist(ax_mz, a1_vals, a2_vals, max_a, zoomed=True, title_font=10.0)
        ax_mz.set_xlabel(rf"{subdet} Harmonic Modulation Amplitude [%]", fontsize=14)
        ax_mz.set_ylabel("Runs", fontsize=14)
        fig_mz.tight_layout()
        fig_mz.savefig(parent / f"modulation_distribution_{subdet}_zoomed.png", dpi=300, bbox_inches="tight")
        plt.close(fig_mz)

        # Events
        fig_e, ax_e = plt.subplots(figsize=(8.5, 6.5))
        _draw_events_hist(ax_e, nevts, min_events, title_font=10.0)
        ax_e.set_xlabel("Total Events per Run", fontsize=14)
        ax_e.set_ylabel("Runs", fontsize=14)
        fig_e.tight_layout()
        fig_e.savefig(parent / f"events_distribution_{subdet}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_e)


def plot_metric_comparison_1x3(
    data_list: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    metric_type: str = "rms",
    max_rms_pct: float = 3.5,
    max_chi2: float = 2.0,
    max_a: float = 1.0,
    min_events: float = 100000.0,
    zoomed: bool = False,
) -> None:
    """
    Generate a 1x3 comparison plot displaying the same metric for South (S),
    North (N), and North+South combined (NS) side by side with consistent axes.
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subdets = ["S", "N", "NS"]
    subdet_titles = ["South Arm (S)", "North Arm (N)", "North+South Combined (NS)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), sharey=True)

    if metric_type == "rms":
        all_vals = [
            [m.get("rms_pct", np.nan) for m in [_get_subdet_metric(d, s) for d in data_list] if not np.isnan(m.get("rms_pct", np.nan))]
            for s in subdets
        ]
        valid_max = [max(v) for v in all_vals if len(v)]
        max_seen = max(valid_max) if valid_max else max_rms_pct
        if zoomed:
            x_max = 2.0
        else:
            x_max = max(max_rms_pct * 1.05, max_seen * 1.05)

        for i, sdet in enumerate(subdets):
            _draw_rms_hist(axes[i], all_vals[i], max_rms_pct, zoomed=zoomed)
            axes[i].set_title(subdet_titles[i], fontsize=14, fontweight="bold")
            axes[i].set_xlabel(rf"{sdet} RMS Non-Flatness [%]", fontsize=13)
            axes[i].set_xlim(0, x_max)
            if i == 0:
                axes[i].set_ylabel("Runs", fontsize=14)

    elif metric_type == "chi2":
        all_vals = [
            [m.get("chi2_ndf", np.nan) for m in [_get_subdet_metric(d, s) for d in data_list] if not np.isnan(m.get("chi2_ndf", np.nan))]
            for s in subdets
        ]
        valid_max = [max(v) for v in all_vals if len(v)]
        max_seen = max(valid_max) if valid_max else max_chi2
        if zoomed:
            x_max = 1.4
        else:
            x_max = max(max_chi2 * 1.05, max_seen * 1.05)

        for i, sdet in enumerate(subdets):
            _draw_chi2_hist(axes[i], all_vals[i], max_chi2, zoomed=zoomed)
            axes[i].set_title(subdet_titles[i], fontsize=14, fontweight="bold")
            axes[i].set_xlabel(rf"{sdet} $\chi^2 / \mathrm{{ndf}}$ vs Flat", fontsize=13)
            axes[i].set_xlim(0, x_max)
            if i == 0:
                axes[i].set_ylabel("Runs", fontsize=14)

    elif metric_type == "modulation":
        all_a1 = [
            [m.get("A1_pct", np.nan) for m in [_get_subdet_metric(d, s) for d in data_list] if not np.isnan(m.get("A1_pct", np.nan))]
            for s in subdets
        ]
        all_a2 = [
            [m.get("A2_pct", np.nan) for m in [_get_subdet_metric(d, s) for d in data_list] if not np.isnan(m.get("A2_pct", np.nan))]
            for s in subdets
        ]
        max_seen = 0.5
        for v in all_a1 + all_a2:
            if len(v):
                max_seen = max(max_seen, max(v))
        if zoomed:
            x_max = 0.8
        else:
            x_max = max(max_a * 1.08, max_seen * 1.08)

        for i, sdet in enumerate(subdets):
            _draw_modulations_hist(axes[i], all_a1[i], all_a2[i], max_a, zoomed=zoomed)
            axes[i].set_title(subdet_titles[i], fontsize=14, fontweight="bold")
            axes[i].set_xlabel(rf"{sdet} Harmonic Modulation Amplitude [%]", fontsize=13)
            axes[i].set_xlim(0, x_max)
            if i == 0:
                axes[i].set_ylabel("Runs", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 4. ENSEMBLE PROFILE ACROSS ALL GOOD RUNS
# ==============================================================================

def plot_ensemble_profile(
    data_list: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    subdet: str = "NS",
) -> None:
    """
    Plot mean and median ratio profile across all good runs to verify collective flatness.
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    good_data = [
        d for d in data_list
        if _get_subdet_metric(d, subdet).get("status") == "GOOD"
        and len(_get_subdet_metric(d, subdet).get("ratios", [])) > 0
    ]

    if not good_data:
        print(f"Warning: No good runs found to compute ensemble profile for {subdet}")
        return

    sample_m = _get_subdet_metric(good_data[0], subdet)
    bin_centers = sample_m["bin_centers"]
    edges = sample_m["edges"]

    ratio_matrix = np.array([_get_subdet_metric(d, subdet)["ratios"] for d in good_data])
    mean_profile = np.mean(ratio_matrix, axis=0)
    median_profile = np.median(ratio_matrix, axis=0)
    std_profile = np.std(ratio_matrix, axis=0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Top panel: Mean & Median ratios with 1-sigma band
    ax_top.plot(bin_centers, mean_profile, color="black", linewidth=2.0, label=rf"Ensemble Mean ({len(good_data):,} Runs)")
    ax_top.plot(bin_centers, median_profile, color="royalblue", linestyle="--", linewidth=1.8, label="Ensemble Median")
    ax_top.fill_between(bin_centers, mean_profile - std_profile, mean_profile + std_profile, color="gray", alpha=0.3, label=r"$\pm 1\sigma$ Run Variation")
    ax_top.axhline(1.0, color="crimson", linestyle=":", linewidth=1.5, label="Perfect Flat (1.0)")

    ax_top.set_ylabel(rf"Ratio to Mean ($\mathrm{{d}}N / \mathrm{{d}}(2\Psi_2) / \langle\mathrm{{d}}N\rangle$)", fontsize=13)

    # Tighten top y-limits adaptively
    max_top_dev = np.max(np.abs(mean_profile - 1.0) + std_profile)
    top_span = max(0.025, float(np.ceil(max_top_dev * 1.35 * 100.0) / 100.0))
    ax_top.set_ylim(1.0 - top_span, 1.0 + top_span)
    ax_top.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9)
    ax_top.grid(True, linestyle="--", alpha=0.4)

    # Bottom panel: Percent deviation from 1.0
    pct_dev = (mean_profile - 1.0) * 100.0
    ax_bot.plot(bin_centers, pct_dev, color="black", linewidth=1.8)
    ax_bot.axhline(0.0, color="crimson", linestyle=":", linewidth=1.5)
    ax_bot.axhspan(-0.5, 0.5, color="forestgreen", alpha=0.15, label=r"$\pm 0.5\%$ Tolerance")
    ax_bot.set_ylabel("Dev [%]", fontsize=13)
    ax_bot.set_xlabel(rf"$2\Psi_2^{{\mathrm{{{subdet}}}}}$ [rad]", fontsize=14)
    ax_bot.set_xlim(-np.pi, np.pi)

    # Tighten bottom y-limits adaptively
    max_pct = np.max(np.abs(pct_dev))
    bot_span = max(0.8, float(np.ceil(max_pct * 1.35 * 10.0) / 10.0))
    ax_bot.set_ylim(-bot_span, bot_span)
    ax_bot.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.9)
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    subdet_label = "North+South Combined" if subdet == "NS" else ("North Arm" if subdet == "N" else "South Arm")
    ax_top.text(
        0.03,
        0.92,
        f"sEPD Flatness QA: {subdet_label}\nGood Runs: {len(good_data):,}",
        transform=ax_top.transAxes,
        fontsize=11.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="gray"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 5. DIAGNOSTIC RUN PLOT (3-ARM COMPARISON WITH FOURIER CURVES)
# ==============================================================================

def plot_diagnostic_run(
    metric_record: Dict[str, Any],
    output_path: Union[str, Path],
    title_prefix: Optional[str] = None,
    show_suptitle: bool = True,
    zoom_ratio: bool = True,
) -> None:
    """
    Generate high-resolution diagnostic plot for a single run showing North, South,
    and Combined North-South side-by-side (Raw Counts and Ratio to Flat with Fourier fit).
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_number = metric_record["run_number"]
    subdets = ["NS", "N", "S"]
    subdet_names = ["North+South (NS)", "North Arm (N)", "South Arm (S)"]

    # Compute maximum deviation across all subdetectors for consistent ratio zoom
    all_devs = []
    for sd in subdets:
        sm = _get_subdet_metric(metric_record, sd)
        if "ratios" in sm and len(sm["ratios"]):
            all_devs.append(np.max(np.abs(np.array(sm["ratios"]) - 1.0)))
    max_run_dev = max(all_devs) if all_devs else 0.02
    run_half_span = max(0.035, float(max_run_dev) * 1.35) if zoom_ratio else 0.10

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex="col")

    for col, (sdet, sname) in enumerate(zip(subdets, subdet_names)):
        m = _get_subdet_metric(metric_record, sdet)
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if m.get("total_events", 0) <= 0 or len(m.get("values", [])) == 0:
            ax_top.text(0.5, 0.5, "NO EVENTS / EMPTY", ha="center", va="center", transform=ax_top.transAxes, fontsize=14, color="crimson")
            ax_bot.text(0.5, 0.5, "NO EVENTS / EMPTY", ha="center", va="center", transform=ax_bot.transAxes, fontsize=14, color="crimson")
            continue

        vals = m["values"]
        mean_bin = m["mean_bin"]
        bin_centers = m["bin_centers"]
        ratios = m["ratios"]
        a1, b1 = m.get("a1", 0.0), m.get("b1", 0.0)
        a2, b2 = m.get("a2", 0.0), m.get("b2", 0.0)

        # 1. Top Panel: Raw Counts
        # Step plot for raw events
        ax_top.step(bin_centers, vals, where="mid", color="navy", linewidth=1.5, label="Data")
        ax_top.axhline(mean_bin, color="crimson", linestyle="--", linewidth=1.5, label=rf"Mean ({mean_bin:,.1f})")
        ax_top.set_title(sname, fontsize=14, fontweight="bold")
        ax_top.set_ylabel("Events / Bin", fontsize=12)
        # 1.55 headroom ensures the info box (top-left) and legend (top-right) never cover data
        max_val = max(vals) if len(vals) else 100.0
        ax_top.set_ylim(bottom=0.0, top=max_val * 1.55)
        ax_top.legend(loc="upper right", ncol=1, fontsize=10.0, frameon=True, framealpha=0.9)
        ax_top.grid(True, linestyle="--", alpha=0.35)

        # Info box in top-left headroom
        status_color = "darkgreen" if m.get("status") == "GOOD" else "crimson"
        info_txt = (
            rf"Run {run_number} [{sdet}]" + "\n" +
            rf"Events: {m['total_events']:,.0f}" + "\n" +
            rf"$\chi^2/\mathrm{{ndf}}$: {m['chi2_ndf']:.2f}, MaxDev: {m.get('max_dev_pct', 0.0):.1f}%" + "\n" +
            rf"RMS: {m['rms_pct']:.2f}% (excess: {m['rms_excess_pct']:.2f}%)" + "\n" +
            rf"$A_1$: {m['A1_pct']:.2f}%, $A_2$: {m['A2_pct']:.2f}%" + "\n" +
            rf"Status: {m['status']}"
        )
        ax_top.text(
            0.03,
            0.96,
            info_txt,
            transform=ax_top.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.92, edgecolor=status_color, linewidth=1.5),
        )

        # 2. Bottom Panel: Ratio to Mean + Fourier Fit Curve
        yerr = np.sqrt(np.where(vals > 0, vals, 1.0)) / mean_bin
        ax_bot.errorbar(bin_centers, ratios, yerr=yerr, fmt="o", markersize=3, color="black", ecolor="gray", elinewidth=1, capsize=0, label="Data Ratio")

        # Smooth Fourier curve
        phi_smooth = np.linspace(-np.pi, np.pi, 300)
        fit_curve = 1.0 + a1 * np.cos(phi_smooth) + b1 * np.sin(phi_smooth) + a2 * np.cos(2.0 * phi_smooth) + b2 * np.sin(2.0 * phi_smooth)
        ax_bot.plot(phi_smooth, fit_curve, color="crimson", linewidth=2.0, label="Fourier Fit ($n=1,2$)")

        ax_bot.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax_bot.axhspan(0.98, 1.02, color="forestgreen", alpha=0.15, label=r"$\pm 2\%$ Band")
        ax_bot.set_xlabel(rf"$2\Psi_2^{{\mathrm{{{sdet}}}}}$ [rad]", fontsize=13)
        ax_bot.set_ylabel(r"Ratio to Mean", fontsize=12)
        ax_bot.set_xlim(-np.pi, np.pi)
        ax_bot.set_ylim(1.0 - run_half_span, 1.0 + run_half_span)
        ax_bot.legend(loc="lower center", ncol=3, fontsize=9.5, frameon=True, framealpha=0.9)
        ax_bot.grid(True, linestyle="--", alpha=0.35)

    if show_suptitle:
        title = f"sEPD Q-Vector Flattening QA: Run {run_number} ({metric_record.get('status', 'UNKNOWN')})"
        if title_prefix:
            title = f"{title_prefix}{title}"
        fig.suptitle(title, fontsize=16, y=0.99)
        fig.tight_layout()
    else:
        fig.tight_layout(pad=0.8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_flat_example(
    metric: Dict[str, Any],
    rank: int,
    output_path: Union[str, Path],
) -> None:
    """
    Generate diagnostic plot for a top flat run example without overall suptitle
    and with zoomed ratio y-axis.
    """
    plot_diagnostic_run(
        metric,
        output_path,
        title_prefix=f"Top Flat #{rank}: ",
        show_suptitle=False,
        zoom_ratio=True,
    )


# ==============================================================================
# 6. FAILURE MODE GALLERY UTILITIES
# ==============================================================================

def plot_failure_example(
    metric: Dict[str, Any],
    failure_mode: str,
    output_path: Union[str, Path],
) -> None:
    """
    Generate diagnostic plot for a failure mode example run with failure mode banner.
    """
    plot_diagnostic_run(metric, output_path)


def plot_failure_mode_metric_distribution(
    metrics_list: Sequence[Dict[str, Any]],
    failure_mode: str,
    example_runs: Sequence[int],
    output_path: Union[str, Path],
    subdet: str = "NS",
    max_rms_pct: float = 3.5,
    max_chi2: float = 2.0,
    max_a: float = 1.0,
) -> None:
    """
    Generate metric distribution highlighting where the selected failure mode example runs lie.
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    m_list = [_get_subdet_metric(d, subdet) for d in metrics_list]

    if "CHI2" in failure_mode:
        vals = [m.get("chi2_ndf", np.nan) for m in m_list if not np.isnan(m.get("chi2_ndf", np.nan))]
        x_max = max(max_chi2 * 1.05, max(vals) * 1.05) if vals else max_chi2 * 1.1
        ax.hist(vals, bins=np.linspace(0, x_max, 45), histtype="step", color="forestgreen", linewidth=2.0)
        ax.axvline(max_chi2, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_chi2:.1f})")
        ax.set_xlim(0, x_max)
        ax.set_xlabel(rf"{subdet} $\chi^2 / \mathrm{{ndf}}$")
        val_key = "chi2_ndf"
    elif "MODULATION" in failure_mode:
        vals = [max(m.get("A1_pct", 0), m.get("A2_pct", 0)) for m in m_list]
        x_max = max(max_a * 1.08, max(vals) * 1.08) if vals else max_a * 1.1
        ax.hist(vals, bins=np.linspace(0, x_max, 45), histtype="step", color="mediumvioletred", linewidth=2.0)
        ax.axvline(max_a, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_a:.1f}%)")
        ax.set_xlim(0, x_max)
        ax.set_xlabel(rf"{subdet} Peak Harmonic Amplitude $\max(A_1, A_2)$ [%]")
        val_key = "max_A"
    elif "LOW_STATS" in failure_mode:
        vals = [m.get("total_events", 0) for m in m_list if m.get("total_events", 0) > 0]
        x_min = max(1.0, min(vals) * 0.85) if vals else 1e4
        x_max = max(vals) * 1.15 if vals else 1e7
        log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 45)
        ax.hist(vals, bins=log_bins, histtype="step", color="coral", linewidth=2.0)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Total Events")
        val_key = "total_events"
    else:  # NON_FLAT or generic
        vals = [m.get("rms_pct", np.nan) for m in m_list if not np.isnan(m.get("rms_pct", np.nan))]
        x_max = max(max_rms_pct * 1.05, max(vals) * 1.05) if vals else max_rms_pct * 1.1
        ax.hist(vals, bins=np.linspace(0, x_max, 45), histtype="step", color="steelblue", linewidth=2.0)
        ax.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct:.1f}%)")
        ax.set_xlim(0, x_max)
        ax.set_xlabel(rf"{subdet} RMS Non-Flatness [%]")
        val_key = "rms_pct"

    # Mark the example runs with vertical lines
    run_dict = {d["run_number"]: _get_subdet_metric(d, subdet) for d in metrics_list}
    colors = ["darkorange", "purple", "magenta", "cyan"]

    for i, r in enumerate(example_runs):
        if r in run_dict:
            rm = run_dict[r]
            if val_key == "max_A":
                v = max(rm.get("A1_pct", 0), rm.get("A2_pct", 0))
            else:
                v = rm.get(val_key, np.nan)

            if not np.isnan(v):
                col = colors[i % len(colors)]
                ax.axvline(v, color=col, linestyle="-", linewidth=2.0, label=f"Run {r} ({v:.2f})")

    ax.set_yscale("log")
    ax.set_ylabel("Runs")
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9)
    ax.set_title(f"Metric Distribution for Failure Mode: {failure_mode}", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 7. Q-VECTOR COMPONENT DIAGNOSTICS & TRENDS (RECENTERING & FLATTENING)
# ==============================================================================

def plot_qvec_components_diagnostic(
    run_number: int,
    comp_dict: Dict[str, Any],
    output_path: Union[str, Path],
    max_recent_dev: float = 1e-3,
    max_twist_dev: float = 1e-3,
    max_ratio_dev: float = 0.02,
) -> None:
    """
    Generate a 3x3 diagnostic grid showing intermediate Q-vector calibrations:
      - Row 0: Recentering <Qx> and <Qy> vs Centrality [%] (expect 0)
      - Row 1: Twisting covariance <Qxy> vs Centrality [%] (expect 0)
      - Row 2: Flattening variance ratio <Qxx>/<Qyy> vs Centrality [%] (expect 1.0)
    Columns: South Arm (S), North Arm (N), North+South Combined (NS).
    """
    _apply_hep_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subdets = ["S", "N", "NS"]
    subdet_titles = ["South Arm (S)", "North Arm (N)", "North+South Combined (NS)"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex="col")

    sdet_dict = comp_dict.get("subdetectors", {})

    for j, (sdet, stitle) in enumerate(zip(subdets, subdet_titles)):
        data = sdet_dict.get(sdet, {})

        # -------------------------------------------------------------
        # Row 0: Recentering <Qx> & <Qy>
        # -------------------------------------------------------------
        ax_r = axes[0, j]
        ax_r.set_title(stitle, fontsize=15, fontweight="bold")
        if data.get("has_recentering"):
            cent = data["cent_centers"]
            qx = data["qx_vals"]
            qy = data["qy_vals"]
            m_qx = data["max_abs_qx"]
            m_qy = data["max_abs_qy"]

            ax_r.plot(cent, qx, color="royalblue", linewidth=1.8, label=r"$\langle Q_x \rangle$")
            ax_r.plot(cent, qy, color="crimson", linestyle="--", linewidth=1.8, label=r"$\langle Q_y \rangle$")
            ax_r.axhline(0.0, color="black", linestyle=":", linewidth=1.2)

            # Adaptive y-limits centered on 0
            max_val = max(1e-4, max(m_qx if not np.isnan(m_qx) else 0, m_qy if not np.isnan(m_qy) else 0) * 1.5)
            ax_r.set_ylim(-max_val, max_val)

            info_text = (
                rf"$\max |\langle Q_x \rangle| = {m_qx:.2e}$" + "\n" +
                rf"$\max |\langle Q_y \rangle| = {m_qy:.2e}$" + "\n" +
                ("PASS (Centroid = 0)" if (m_qx <= max_recent_dev and m_qy <= max_recent_dev) else "FAIL (Non-zero)")
            )
            border_col = "forestgreen" if (m_qx <= max_recent_dev and m_qy <= max_recent_dev) else "crimson"
            ax_r.text(0.97, 0.95, info_text, transform=ax_r.transAxes, ha="right", va="top", fontsize=9.5,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=border_col, alpha=0.9))
            ax_r.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)
        else:
            # Composite NS is centered via N and S
            ax_r.text(0.5, 0.5, "Composite Vector:\n" + r"$\vec{Q}_{\mathrm{NS}} = \vec{Q}_{\mathrm{N}} + \vec{Q}_{\mathrm{S}}$" + "\n(Centered via N & S arms)",
                      transform=ax_r.transAxes, ha="center", va="center", fontsize=11, color="gray",
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", edgecolor="gray", alpha=0.9))
            ax_r.set_ylim(-1e-4, 1e-4)

        if j == 0:
            ax_r.set_ylabel(r"Recentering $\langle Q_{x,y} \rangle$", fontsize=13)
        ax_r.grid(True, linestyle="--", alpha=0.35)

        # -------------------------------------------------------------
        # Row 1: Twisting Covariance <Qxy>
        # -------------------------------------------------------------
        ax_t = axes[1, j]
        if data.get("has_twisting"):
            cent_xy = data["cent_centers_xy"]
            qxy = data["qxy_vals"]
            m_qxy = data["max_abs_qxy"]

            ax_t.plot(cent_xy, qxy, color="darkviolet", linewidth=1.8, label=r"$\langle Q_{xy} \rangle$")
            ax_t.axhline(0.0, color="black", linestyle=":", linewidth=1.2)

            max_val = max(1e-4, (m_qxy if not np.isnan(m_qxy) else 0) * 1.5)
            ax_t.set_ylim(-max_val, max_val)

            info_text = (
                rf"$\max |\langle Q_{{xy}} \rangle| = {m_qxy:.2e}$" + "\n" +
                ("PASS (Cov = 0)" if m_qxy <= max_twist_dev else "FAIL (Non-zero)")
            )
            border_col = "forestgreen" if m_qxy <= max_twist_dev else "crimson"
            ax_t.text(0.97, 0.95, info_text, transform=ax_t.transAxes, ha="right", va="top", fontsize=9.5,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=border_col, alpha=0.9))
            ax_t.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)
        if j == 0:
            ax_t.set_ylabel(r"Twisting $\langle Q_{xy} \rangle$", fontsize=13)
        ax_t.grid(True, linestyle="--", alpha=0.35)

        # -------------------------------------------------------------
        # Row 2: Flattening Variance Ratio <Qxx> / <Qyy>
        # -------------------------------------------------------------
        ax_f = axes[2, j]
        if data.get("has_flattening_ratio"):
            cent_rat = data["cent_centers_rat"]
            ratios = data["qxx_yy_ratios"]
            m_dev = data["max_dev_ratio_xxyy"]

            ax_f.plot(cent_rat, ratios, color="forestgreen", marker="o", markersize=3, linewidth=1.5, label=r"$\langle Q_{xx} \rangle / \langle Q_{yy} \rangle$")
            ax_f.axhline(1.0, color="crimson", linestyle=":", linewidth=1.5, label="Target (1.0)")
            ax_f.axhspan(0.99, 1.01, color="forestgreen", alpha=0.15, label=r"$\pm 1\%$ Band")

            max_span = max(0.02, (m_dev if not np.isnan(m_dev) else 0) * 1.5)
            ax_f.set_ylim(1.0 - max_span, 1.0 + max_span)

            info_text = (
                rf"$\max |R - 1| = {m_dev:.2e}$" + "\n" +
                ("PASS (Ratio = 1)" if m_dev <= max_ratio_dev else "FAIL (Eccentric)")
            )
            border_col = "forestgreen" if m_dev <= max_ratio_dev else "crimson"
            ax_f.text(0.97, 0.95, info_text, transform=ax_f.transAxes, ha="right", va="top", fontsize=9.5,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=border_col, alpha=0.9))
            ax_f.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)

        if j == 0:
            ax_f.set_ylabel(r"Ratio $\langle Q_{xx} \rangle / \langle Q_{yy} \rangle$", fontsize=13)
        ax_f.set_xlabel("Centrality [%]", fontsize=14)
        ax_f.grid(True, linestyle="--", alpha=0.35)

    title = f"sEPD Intermediate Q-Vector Calibration QA: Run {run_number}"
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_components_trends(
    data_list: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    max_recent_dev: float = 1e-3,
    max_twist_dev: float = 1e-3,
    max_ratio_dev: float = 0.02,
) -> None:
    """
    Generate aggregate trend plots of Q-vector component QA vs Run Number:
      1. trends_components_recentering.png: max |<Qx>| and max |<Qy>| vs Run Number (N and S)
      2. trends_components_flattening.png: max |<Qxy>| and max |<Qxx>/<Qyy> - 1| vs Run Number (S, N, NS)
    """
    _apply_hep_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [d["run_number"] for d in data_list]
    if not runs:
        return

    # Sort by run number
    sorted_idx = np.argsort(runs)
    runs = np.array(runs)[sorted_idx]
    sorted_data = [data_list[i] for i in sorted_idx]

    # 1. Recentering Trends
    fig_rec, axes_rec = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for i, sdet in enumerate(["N", "S"]):
        ax = axes_rec[i]
        qx_max = np.array([d.get(f"Q_{sdet}_x_max_abs", np.nan) for d in sorted_data])
        qy_max = np.array([d.get(f"Q_{sdet}_y_max_abs", np.nan) for d in sorted_data])
        ax.plot(runs, qx_max, label=rf"$\max |\langle Q_x \rangle|$ ({sdet})", color="royalblue", alpha=0.85, linewidth=1.2, marker="o", markersize=3.5)
        ax.plot(runs, qy_max, label=rf"$\max |\langle Q_y \rangle|$ ({sdet})", color="crimson", alpha=0.85, linewidth=1.2, marker="s", markersize=3.5)
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.axhline(max_recent_dev, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label=rf"Cut ({max_recent_dev:.0e})")
        ax.set_ylabel(rf"{sdet} Arm Max Dev", fontsize=12)
        ax.legend(loc="upper right", fontsize=10.0, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(-1e-4, max(max_recent_dev * 1.2, 1e-3))
    axes_rec[1].set_xlabel("Run Number", fontsize=13)
    axes_rec[1].ticklabel_format(style="plain", useOffset=False, axis="x")
    axes_rec[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig_rec.suptitle("sEPD Q-Vector Recentering QA vs Run Number (Target = 0)", fontsize=15, y=0.99)
    fig_rec.tight_layout()
    fig_rec.savefig(output_dir / "trends_components_recentering.png", dpi=300, bbox_inches="tight")
    plt.close(fig_rec)

    # 2. Flattening Trends (Twisting & Ratio)
    fig_flat, axes_flat = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    colors = {"S": "royalblue", "N": "mediumvioletred", "NS": "forestgreen"}
    markers = {"S": "o", "N": "s", "NS": "^"}
    for sdet in ["S", "N", "NS"]:
        qxy_max = np.array([d.get(f"Q_{sdet}_xy_max_abs", np.nan) for d in sorted_data])
        axes_flat[0].plot(runs, qxy_max, label=f"{sdet} Arm", color=colors[sdet], alpha=0.85, linewidth=1.2, marker=markers[sdet], markersize=3.5)

        ratio_dev = np.array([d.get(f"Q_{sdet}_xxyy_max_dev", np.nan) for d in sorted_data])
        axes_flat[1].plot(runs, ratio_dev, label=f"{sdet} Arm", color=colors[sdet], alpha=0.85, linewidth=1.2, marker=markers[sdet], markersize=3.5)

    axes_flat[0].axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    axes_flat[0].axhline(max_twist_dev, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label=rf"Cut ({max_twist_dev:.0e})")
    axes_flat[0].set_ylabel(r"$\max |\langle Q_{xy} \rangle|$", fontsize=12)
    axes_flat[0].legend(loc="upper right", fontsize=10.0, framealpha=0.9)
    axes_flat[0].grid(True, linestyle="--", alpha=0.35)
    axes_flat[0].set_ylim(-1e-4, max(max_twist_dev * 1.2, 1e-3))

    axes_flat[1].axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    axes_flat[1].axhline(max_ratio_dev, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label=rf"Cut ({max_ratio_dev * 100:.0f}%)")
    axes_flat[1].set_ylabel(r"$\max |\langle Q_{xx} \rangle / \langle Q_{yy} \rangle - 1|$", fontsize=12)
    axes_flat[1].set_xlabel("Run Number", fontsize=13)
    axes_flat[1].ticklabel_format(style="plain", useOffset=False, axis="x")
    axes_flat[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes_flat[1].legend(loc="upper right", fontsize=10.0, framealpha=0.9)
    axes_flat[1].grid(True, linestyle="--", alpha=0.35)
    axes_flat[1].set_ylim(-1e-3, max(max_ratio_dev * 1.2, 0.02))

    fig_flat.suptitle("sEPD Q-Vector Flattening QA vs Run Number (Twisting = 0, Ratio = 1)", fontsize=15, y=0.99)
    fig_flat.tight_layout()
    fig_flat.savefig(output_dir / "trends_components_flattening.png", dpi=300, bbox_inches="tight")
    plt.close(fig_flat)

