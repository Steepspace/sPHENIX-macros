#!/usr/bin/env python3
"""
plot_qvec_aggregate.py

Main CLI driver for run-by-run sEPD Q-Vector flatness QA.
Processes 2,000+ calibration ROOT files in parallel, computes statistical and
systematic flatness metrics, writes CSV summaries & good/bad run lists, and
generates publication-quality aggregate QA plots:
- 2D Ratio Heatmaps (Run Index vs 2Psi2)
- 4-Panel Trend Plots vs Run Number
- Metric Distribution Histograms
- Ensemble Flatness Profiles
- Failure Mode Diagnostic Galleries
"""

import argparse
import concurrent.futures
import csv
import functools
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import tqdm

# Ensure local imports from the same directory work reliably
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from qvec_qa_metrics import (
    DEFAULT_CENT_SLICES,
    add_cut_args,
    cuts_from_args,
    evaluate_run,
    extract_qvec_run_metrics,
    parse_cent_slices,
    slice_label,
)
from qvec_qa_plots import (
    plot_heatmap,
    plot_trends,
    plot_metric_distributions,
    plot_metric_comparison_1x3,
    plot_ensemble_profile,
    plot_diagnostic_run,
    plot_top_flat_example,
    plot_failure_example,
    plot_failure_mode_metric_distribution,
    plot_components_trends,
    plot_slice_summary,
)


def _fmt(value: Any, spec: str) -> str:
    return "nan" if value is None or np.isnan(value) else format(value, spec)


SUBDET_CSV_FIELDS = [
    ("total_events", ".0f"),
    ("chi2_ndf", ".3f"),
    ("rms_pct", ".3f"),
    ("rms_sigma", ".3f"),
    ("rms_excess_pct", ".3f"),
    ("max_dev_pct", ".2f"),
    ("max_dev_sigma", ".2f"),
    ("A1_pct", ".3f"),
    ("A1_sig", ".2f"),
    ("A2_pct", ".3f"),
    ("A2_sig", ".2f"),
]


def write_summary_reports(
    data_list: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    subdetectors: Sequence[str] = ("NS", "N", "S"),
) -> None:
    """
    Write summary CSV, good runs list, and flagged outlier runs list.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "qvec_qa_summary.csv"
    flagged_list_path = output_dir / "flagged_outlier_runs.list"
    good_list_path = output_dir / "good_runs.list"

    fieldnames = [
        "run_number",
        "total_events",
        "primary_status",
        "overall_status",
        "slice_status",
    ]

    for sdet in subdetectors:
        fieldnames.extend(f"{k}_{sdet}" for k, _ in SUBDET_CSV_FIELDS if k != "total_events")
        fieldnames.append(f"status_{sdet}")

    comp_fieldnames = [
        "Q_N_x_max_abs",
        "Q_N_y_max_abs",
        "Q_S_x_max_abs",
        "Q_S_y_max_abs",
        "Q_N_xy_max_abs",
        "Q_S_xy_max_abs",
        "Q_NS_xy_max_abs",
        "Q_N_xxyy_max_dev",
        "Q_S_xxyy_max_dev",
        "Q_NS_xxyy_max_dev",
    ]
    fieldnames.extend(comp_fieldnames)

    fieldnames.append("file_path")

    with open(csv_path, "w", newline="") as f_csv, \
         open(flagged_list_path, "w") as f_bad, \
         open(good_list_path, "w") as f_good:

        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for d in data_list:
            row: Dict[str, Any] = {
                "run_number": d["run_number"],
                "total_events": f"{d['total_events']:.0f}",
                "primary_status": d["primary_status"],
                "overall_status": d["status"],
                "slice_status": d.get("slice_status", "not evaluated"),
                "file_path": d["file_path"],
            }

            subdets_dict = d.get("subdetectors", {})
            for sdet in subdetectors:
                sm = subdets_dict.get(sdet, {})
                for k, spec in SUBDET_CSV_FIELDS:
                    if k != "total_events":
                        row[f"{k}_{sdet}"] = _fmt(sm.get(k, np.nan), spec)
                row[f"status_{sdet}"] = sm.get("status", "UNKNOWN")

            for ckey in comp_fieldnames:
                val = d.get(ckey, np.nan)
                row[ckey] = f"{val:.3e}" if not np.isnan(val) else "nan"

            writer.writerow(row)

            # Determine whether run qualifies for good_runs.list
            # Default criterion: primary (NS) status is GOOD
            if d["primary_status"] == "GOOD":
                f_good.write(f"{d['run_number']}\n")
            else:
                f_bad.write(f"{d['run_number']}\t{d['status']}\n")

    print(f"Summary CSV written to: {csv_path}")
    print(f"Good runs list written to: {good_list_path}")
    print(f"Flagged outlier runs list written to: {flagged_list_path}")


def write_slice_reports(
    data_list: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    subdetectors: Sequence[str] = ("NS", "N", "S"),
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Writes qvec_qa_slices.csv (one row per run, centrality slice and subdetector) and
    slice_flagged_runs.list (runs with any slice flag, with the flags). Returns
    counts[slice][subdet][flag] of runs per flag, over runs whose integrated (NS) status is GOOD.
    """
    output_dir = Path(output_dir)
    csv_path = output_dir / "qvec_qa_slices.csv"
    list_path = output_dir / "slice_flagged_runs.list"
    counts: Dict[str, Dict[str, Dict[str, int]]] = {}

    fieldnames = ["run_number", "slice", "subdet"] + [k for k, _ in SUBDET_CSV_FIELDS] + ["status", "primary_status"]
    with open(csv_path, "w", newline="") as f_csv, open(list_path, "w") as f_list:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for d in data_list:
            for label, slice_subdets in d.get("slices", {}).items():
                for sdet in subdetectors:
                    sm = slice_subdets.get(sdet)
                    if sm is None:
                        continue
                    row = {"run_number": d["run_number"], "slice": label, "subdet": sdet,
                           "status": sm.get("status", "UNKNOWN"), "primary_status": d["primary_status"]}
                    row.update({k: _fmt(sm.get(k, np.nan), spec) for k, spec in SUBDET_CSV_FIELDS})
                    writer.writerow(row)
                    if d["primary_status"] == "GOOD":
                        for flag in sm.get("status", "GOOD").split(";"):
                            if flag not in ("GOOD", "FLAG_LOW_STATS", "FLAG_EMPTY_OR_ZERO"):
                                c = counts.setdefault(label, {}).setdefault(sdet, {})
                                c[flag] = c.get(flag, 0) + 1
            if d.get("slice_status", "GOOD") not in ("GOOD", "not evaluated"):
                f_list.write(f"{d['run_number']}\t{d['slice_status']}\n")

    print(f"Centrality-slice CSV written to: {csv_path}")
    print(f"Slice-flagged runs list written to: {list_path}")
    return counts


def is_empty_run(d: Dict[str, Any]) -> bool:
    """True if every subdetector of the run is FLAG_EMPTY_OR_ZERO (no events, nothing to plot)."""
    subdets = d.get("subdetectors", {})
    return bool(subdets) and all(
        "FLAG_EMPTY_OR_ZERO" in sm.get("status", "").split(";") for sm in subdets.values()
    )


def generate_failure_mode_examples(
    metrics_list: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    subdet: str = "NS",
    example_runs_per_mode: int = 3,
    cuts: Optional[Dict[str, float]] = None,
) -> Dict[str, List[int]]:
    """
    Identify representative outlier runs for each failure mode and generate
    diagnostic 3-arm plots and metric distributions.
    """
    output_dir = Path(output_dir)
    examples_dir = output_dir / "failure_examples"
    cuts = cuts or {}
    max_a = cuts.get("max_a_ns", 0.75) if subdet == "NS" else cuts.get("max_a_arm", 1.0)

    failure_modes_config = [
        {
            "mode": "FLAG_HIGH_CHI2",
            "description": f"High Flatness Chi2/ndf (> {cuts.get('max_chi2', 2.0):g})",
            "filter_fn": lambda d: "FLAG_HIGH_CHI2" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: d.get("subdetectors", {}).get(subdet, {}).get("chi2_ndf", 0.0),
            "reverse": True,
        },
        {
            "mode": "FLAG_NON_FLAT",
            "description": "Significant Non-Flatness (RMS or single-bin deviation)",
            "filter_fn": lambda d: "FLAG_NON_FLAT" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: d.get("subdetectors", {}).get(subdet, {}).get("rms_pct", 0.0),
            "reverse": True,
        },
        {
            "mode": "FLAG_RESIDUAL_MODULATION",
            "description": f"Residual Harmonic Modulation (> {max_a:g}% and significant)",
            "filter_fn": lambda d: "FLAG_RESIDUAL_MODULATION" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: max(
                d.get("subdetectors", {}).get(subdet, {}).get("A1_pct", 0.0),
                d.get("subdetectors", {}).get(subdet, {}).get("A2_pct", 0.0),
            ),
            "reverse": True,
        },
        {
            "mode": "FLAG_LOW_STATS",
            "description": "Low Event Statistics",
            "filter_fn": lambda d: "FLAG_LOW_STATS" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: d.get("subdetectors", {}).get(subdet, {}).get("total_events", 0.0),
            "reverse": False,
        },
        {
            "mode": "FLAG_EMPTY_OR_ZERO",
            "description": "Zero Events or Missing Histogram",
            "filter_fn": lambda d: "FLAG_EMPTY_OR_ZERO" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: d.get("run_number", 0),
            "reverse": False,
        },
    ]

    examples_map: Dict[str, List[int]] = {}
    used_runs = set()

    for cfg in failure_modes_config:
        mode = cfg["mode"]
        filter_fn = cfg["filter_fn"]
        matching_runs = [d for d in metrics_list if filter_fn(d)]

        if not matching_runs:
            examples_map[mode] = []
            continue

        # Sort matching runs by severity
        sorted_runs = sorted(
            matching_runs,
            key=lambda d: (d["run_number"] in used_runs, -cfg["sort_key"](d) if cfg["reverse"] else cfg["sort_key"](d))
        )
        selected = sorted_runs[:example_runs_per_mode]
        examples_map[mode] = [d["run_number"] for d in selected]

        for d in selected:
            used_runs.add(d["run_number"])

        # Runs without events have no distributions to show; list them only
        if mode == "FLAG_EMPTY_OR_ZERO":
            continue

        mode_dir = examples_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for d in selected:
            plot_path = mode_dir / f"run_{d['run_number']}_diagnostic.png"
            plot_failure_example(d, failure_mode=mode, output_path=plot_path)

        # Plot metric distribution highlighting the example runs
        dist_plot_path = mode_dir / f"metric_distribution_{mode}.png"
        plot_failure_mode_metric_distribution(
            metrics_list=metrics_list,
            failure_mode=mode,
            example_runs=[d["run_number"] for d in selected],
            output_path=dist_plot_path,
            subdet=subdet,
            rms_cut=cuts.get("max_rms_sigma", 1.3),
            max_chi2=cuts.get("max_chi2", 2.0),
            max_a=max_a,
        )

    return examples_map


def generate_top_flat_examples(
    metrics_list: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    subdet: str = "NS",
    n_examples: int = 3,
    min_events: float = 500000.0,
) -> List[int]:
    """
    Select the top n_examples flattest runs (with sufficient statistics)
    and generate diagnostic 3-arm plots for reference.
    """
    output_dir = Path(output_dir)
    top_dir = output_dir / "top_flat_examples"
    top_dir.mkdir(parents=True, exist_ok=True)

    # Filter for good runs with sufficient events
    good_runs = [
        d for d in metrics_list
        if d.get("primary_status") == "GOOD"
        and d.get("total_events", 0) >= min_events
        and not np.isnan(d.get("subdetectors", {}).get(subdet, {}).get("chi2_ndf", np.nan))
    ]

    if not good_runs:
        good_runs = [
            d for d in metrics_list
            if d.get("primary_status") == "GOOD"
            and not np.isnan(d.get("subdetectors", {}).get(subdet, {}).get("chi2_ndf", np.nan))
        ]

    # Sort by lowest chi2_ndf on subdet
    sorted_runs = sorted(
        good_runs,
        key=lambda d: d.get("subdetectors", {}).get(subdet, {}).get("chi2_ndf", 999.0)
    )

    selected = sorted_runs[:n_examples]
    selected_run_numbers = []

    for rank, d in enumerate(selected, start=1):
        r_num = d["run_number"]
        selected_run_numbers.append(r_num)
        plot_path = top_dir / f"run_{r_num}_rank_{rank}_diagnostic.png"
        plot_top_flat_example(d, rank=rank, output_path=plot_path)

    return selected_run_numbers


def explain_arm_flags(arm_label: str, sm: Dict[str, Any], cuts: Dict[str, float], subdet: str) -> List[str]:
    """
    Explain each flag in one arm's status using that arm's own metrics and the cuts it was evaluated with.
    """
    flags = [f for f in sm.get("status", "GOOD").split(";") if f and f != "GOOD"]
    if not flags:
        return []

    nevt = sm.get("total_events", 0.0)
    mean_bin = sm.get("mean_bin", 0.0)
    chi2 = sm.get("chi2_ndf", np.nan)
    rms_sig = sm.get("rms_sigma", np.nan)
    rms_ex = sm.get("rms_excess_pct", np.nan)
    max_dev = sm.get("max_dev_pct", np.nan)
    max_dev_sig = sm.get("max_dev_sigma", np.nan)
    a1, a1_sig = sm.get("A1_pct", np.nan), sm.get("A1_sig", np.nan)
    a2, a2_sig = sm.get("A2_pct", np.nan), sm.get("A2_sig", np.nan)
    max_a = cuts["max_a_ns"] if subdet == "NS" else cuts["max_a_arm"]

    reasons = []
    for flag in flags:
        if flag == "FLAG_EMPTY_OR_ZERO":
            text = "no events or missing histogram"
        elif flag == "FLAG_LOW_STATS":
            text = f"{nevt:,.0f} events (cut: {cuts['min_events']:,.0f})"
        elif flag == "FLAG_HIGH_CHI2":
            text = f"chi2/ndf vs flat = {chi2:.2f} (cut: {cuts['max_chi2']:g})"
        elif flag == "FLAG_NON_FLAT":
            parts = []
            if rms_sig > cuts["max_rms_sigma"] and rms_ex > cuts["max_excess_rms"]:
                parts.append(
                    f"RMS {rms_sig:.2f}x the Poisson expectation (cut: {cuts['max_rms_sigma']:g}) "
                    f"with excess RMS {rms_ex:.2f}% (cut: {cuts['max_excess_rms']:g}%)"
                )
            if max_dev_sig > cuts["max_dev_sigma"]:
                parts.append(
                    f"one bin {max_dev:.2f}% from flat = {max_dev_sig:.1f} Poisson sigma "
                    f"(cut: {cuts['max_dev_sigma']:g}; ~{mean_bin:,.0f} events/bin)"
                )
            text = "; ".join(parts) if parts else "non-flat"
            text += f". chi2/ndf {chi2:.2f}, A2 {a2:.2f}%"
        elif flag == "FLAG_RESIDUAL_MODULATION":
            text = (f"A1 {a1:.2f}% ({a1_sig:.1f} sigma), A2 {a2:.2f}% ({a2_sig:.1f} sigma) "
                    f"(cut: {max_a:g}% and {cuts['min_a_sig']:g} sigma)")
        else:
            text = ""
        reasons.append(f"**{arm_label}** `{flag}`" + (f": {text}" if text else ""))
    return reasons


def generate_flagged_arm_examples(
    metrics_list: Sequence[Dict[str, Any]],
    output_dir: Path,
    cuts: Dict[str, float],
) -> List[int]:
    """
    Generate diagnostic plots for all runs that have any flagged detector arm status
    (overall status != 'GOOD' or any subdetector status != 'GOOD') into a dedicated
    'flagged_arm_runs' subfolder. Also writes a summary markdown report explaining
    the specific physical and statistical root causes for each flag.
    """
    flagged_dir = output_dir / "flagged_arm_runs"
    flagged_dir.mkdir(parents=True, exist_ok=True)

    all_flagged = [
        d for d in metrics_list
        if d.get("status") != "GOOD"
        or any(sm.get("status") != "GOOD" for sm in d.get("subdetectors", {}).values())
    ]
    # Runs with no events in any arm would only give empty plots: list them, don't plot them
    empty_runs = sorted(d["run_number"] for d in all_flagged if is_empty_run(d))
    flagged_runs = [d for d in all_flagged if not is_empty_run(d)]
    flagged_run_numbers = []

    readme_lines = [
        "# Flagged Arm Runs Investigation Summary\n",
        "This directory contains detailed diagnostic plots for runs where one or more detector arms",
        "were flagged with a non-GOOD status during Q-vector flatness QA.\n",
        f"- **Total Runs Analyzed:** {len(metrics_list):,}",
        f"- **Flagged Runs Identified:** {len(all_flagged)}",
        f"- **Flagged Runs Plotted:** {len(flagged_runs)}",
        f"- **Empty Runs (FLAG_EMPTY_OR_ZERO in every arm, not plotted):** {len(empty_runs)}\n",
        "| Run Number | Overall Status | Primary (NS) | North Arm (N) | South Arm (S) | Events | Root Cause / Detailed Explanation |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for d in flagged_runs:
        r_num = d["run_number"]
        flagged_run_numbers.append(r_num)
        status_overall = d.get("status", "UNKNOWN")
        subdets = d.get("subdetectors", {})
        ns_s = subdets.get("NS", {})
        n_s = subdets.get("N", {})
        s_s = subdets.get("S", {})

        # Explain each arm's flags with its own metrics
        reasons = []
        for arm_label, sm, sdet in [("South Arm (S)", s_s, "S"), ("North Arm (N)", n_s, "N"), ("Combined (NS)", ns_s, "NS")]:
            reasons.extend(explain_arm_flags(arm_label, sm, cuts, sdet))

        # A single arm flagged while the combined NS is GOOD usually means the arms' residuals cancel
        if ns_s.get("status") == "GOOD" and (s_s.get("status") != "GOOD" or n_s.get("status") != "GOOD"):
            reasons.append(
                f"Combined NS is GOOD (chi2/ndf {ns_s.get('chi2_ndf', np.nan):.2f}, A2 {ns_s.get('A2_pct', np.nan):.2f}%)"
            )

        for fl in status_overall.split(";"):
            if "FLAG_RECENT" in fl:
                reasons.append(f"Non-zero recentering detected: `{fl}`")
            elif "FLAG_TWIST" in fl:
                reasons.append(f"Non-zero twisting covariance detected: `{fl}`")
            elif "FLAG_FLAT_ECCENTRICITY" in fl:
                reasons.append(f"Flattening variance ratio deviation: `{fl}`")

        reason_str = "<br>".join(reasons) if reasons else "Threshold exceeded"

        readme_lines.append(
            f"| [{r_num}](run_{r_num}_diagnostic.png) | `{status_overall}` | `{ns_s.get('status')}` | `{n_s.get('status')}` | `{s_s.get('status')}` | {d.get('total_events', 0):,.0f} | {reason_str} |"
        )

        plot_path = flagged_dir / f"run_{r_num}_diagnostic.png"
        plot_diagnostic_run(
            d,
            plot_path,
            title_prefix=f"Flagged Run {r_num} ({status_overall}): ",
            show_suptitle=False,
            zoom_ratio=True,
        )

    if empty_runs:
        readme_lines.extend([
            "",
            f"## Empty Runs ({len(empty_runs)})\n",
            "No events in any arm (zero events or missing histogram), so no diagnostic plot is made.\n",
            ", ".join(str(r) for r in empty_runs),
        ])

    with open(flagged_dir / "README.md", "w") as f_rm:
        f_rm.write("\n".join(readme_lines) + "\n")

    return flagged_run_numbers


def reevaluate_data_list(data_list: List[Dict[str, Any]], cuts: Dict[str, float]) -> None:
    """Re-evaluates all QA statuses of cached run records with the current cuts, without re-reading ROOT files."""
    for d in data_list:
        evaluate_run(d, cuts)


def extract_runs(
    file_paths: Sequence[str],
    args: argparse.Namespace,
    cuts: Dict[str, float],
    cent_slices: Sequence[tuple],
) -> List[Dict[str, Any]]:
    """Reads the QA metrics of every file in parallel, sorted by run number."""
    extract_fn = functools.partial(
        extract_qvec_run_metrics,
        subdetectors=("NS", "N", "S"),
        harmonic=args.harmonic,
        pass_suffix="_corr2",
        cent_min=args.cent_min,
        cent_max=args.cent_max,
        cent_slices=cent_slices,
        check_components=not args.no_components,
        cuts=cuts,
    )

    data_list: List[Dict[str, Any]] = []
    errors: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_fn, fp): fp for fp in file_paths}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting Metrics"):
            res, err = future.result()
            if res is not None:
                data_list.append(res)
            else:
                errors.append(err or "Unknown error")

    if errors:
        print(f"Warning: {len(errors)} files failed to process. First 5 errors:")
        for e in errors[:5]:
            print(f"  - {e}")

    data_list.sort(key=lambda d: d["run_number"])
    return data_list


def main():
    parser = argparse.ArgumentParser(
        description="Run-by-run sEPD Q-Vector flatness QA aggregate suite."
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--file-list", "-l",
        type=str,
        default=None,
        help="Path to a text file containing ROOT file paths (one per line).",
    )
    input_group.add_argument(
        "--files", "-f",
        nargs="+",
        default=None,
        help="List of ROOT file paths or wildcards.",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="qvec_qa_output",
        help="Directory to store output plots and summary reports (default: qvec_qa_output).",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=16,
        help="Number of parallel worker threads (default: 16).",
    )
    parser.add_argument(
        "--subdetector",
        type=str,
        default="all",
        choices=["all", "NS", "N", "S"],
        help="Subdetector to analyze/plot: all, NS, N, S (default: all).",
    )
    parser.add_argument(
        "--harmonic",
        type=int,
        default=2,
        help="Event-plane harmonic order n (default: 2 for 2Psi2).",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        nargs="+",
        default=[],
        help="Specific run numbers to highlight on plots.",
    )

    # QA cut thresholds (shared with plot_qvec_run.py)
    add_cut_args(parser)
    parser.add_argument(
        "--top-flat-examples",
        type=int,
        default=3,
        help="Number of representative top flattest runs to save for reference (default: 3).",
    )
    parser.add_argument(
        "--no-components",
        action="store_true",
        help="Disable intermediate Q-vector component QA checks (recentering, twisting, variance ratio).",
    )

    # Caching options
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable pickle caching of extracted run metrics (loads if cache exists, saves if not).",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Custom path to the pickle cache file (default: <output-dir>/qvec_qa_cache.pkl).",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-reading of input ROOT files even if cache exists.",
    )

    # Centrality
    parser.add_argument(
        "--cent-min",
        type=float,
        default=None,
        help="Minimum centrality [%%] of the main (integrated) projection (default: all).",
    )
    parser.add_argument(
        "--cent-max",
        type=float,
        default=None,
        help="Maximum centrality [%%] of the main (integrated) projection (default: all).",
    )
    parser.add_argument(
        "--cent-slices",
        type=str,
        default=",".join(slice_label(s) for s in DEFAULT_CENT_SLICES),
        help="Centrality slices [%%] evaluated in addition to the integrated distribution, e.g. '0-20,60-80'; "
             "'none' to disable. Slice flags are reported (qvec_qa_slices.csv, slice_flagged_runs.list) but do not "
             f"change the good-run list. Default: {','.join(slice_label(s) for s in DEFAULT_CENT_SLICES)}.",
    )

    parser.add_argument(
        "--example-runs-per-mode",
        type=int,
        default=3,
        help="Number of representative runs to plot per failure mode (default: 3).",
    )

    args = parser.parse_args()
    cuts = cuts_from_args(args)
    cent_slices = parse_cent_slices(args.cent_slices)
    slice_labels = [slice_label(s) for s in cent_slices]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subdetectors = ("NS", "N", "S")

    use_cache = args.cache or (args.cache_file is not None)
    cache_path = Path(args.cache_file) if args.cache_file else (output_dir / "qvec_qa_cache.pkl")

    print(f"============================================================")
    print(f"  sEPD Q-Vector Run-by-Run QA Aggregate Suite")
    print(f"  Output directory: {output_dir}")
    print(f"  Cuts: " + ", ".join(f"{k}={v:g}" for k, v in cuts.items()))
    print(f"  Centrality slices: {', '.join(slice_labels) if slice_labels else 'none'}")
    print(f"============================================================")

    data_list: List[Dict[str, Any]] = []
    loaded_from_cache = False
    if use_cache and cache_path.exists() and not args.force_reload:
        print(f"Loading cached metrics from: {cache_path}")
        try:
            with open(cache_path, "rb") as f_pkl:
                data_list = pickle.load(f_pkl)
            print(f"Successfully loaded {len(data_list):,} runs from cache.")
            loaded_from_cache = True
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}. Falling back to ROOT files.")

    if loaded_from_cache:
        cached_slices = set(data_list[0].get("slices", {}).keys()) if data_list else set()
        if not set(slice_labels) <= cached_slices:
            # The cache predates (or uses other) centrality slices: re-read the files it lists
            print(f"Cache has centrality slices {sorted(cached_slices) or 'none'}; re-reading the "
                  f"{len(data_list):,} cached files for slices {slice_labels}.")
            data_list = extract_runs([d["file_path"] for d in data_list], args, cuts, cent_slices)
            loaded_from_cache = False
        else:
            reevaluate_data_list(data_list, cuts)
    else:
        file_paths: List[str] = []
        if args.file_list:
            list_path = Path(args.file_list)
            if not list_path.exists():
                print(f"Error: File list not found: {list_path}")
                sys.exit(1)
            with open(list_path) as f:
                file_paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        elif args.files:
            file_paths = args.files
        else:
            print(f"Error: No inputs specified. Provide --file-list, --files, or specify an existing --cache-file.")
            sys.exit(1)

        print(f"  Total input files: {len(file_paths):,} | Harmonic: {args.harmonic} | Parallel workers: {args.workers}")
        data_list = extract_runs(file_paths, args, cuts, cent_slices)

    if not data_list:
        print("Error: No valid run data extracted.")
        sys.exit(1)

    if use_cache and not loaded_from_cache:
        print(f"\nSaving {len(data_list):,} runs to pickle cache: {cache_path} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f_pkl:
            pickle.dump(data_list, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cache saved successfully ({cache_path.stat().st_size / (1024*1024):.1f} MB).")

    # 1. Summary CSV and Run Lists
    write_summary_reports(data_list, output_dir, subdetectors=subdetectors)

    # Summary statistics to console
    total_runs = len(data_list)
    good_runs = sum(1 for d in data_list if d["primary_status"] == "GOOD")
    flagged_runs = total_runs - good_runs
    print(f"\nQA Summary:")
    print(f"  Total runs analyzed: {total_runs:,}")
    print(f"  Good runs (NS):      {good_runs:,} ({100.0 * good_runs / total_runs:.1f}%)")
    print(f"  Flagged runs (NS):   {flagged_runs:,} ({100.0 * flagged_runs / total_runs:.1f}%)")

    # Centrality slices: reported, but do not change the good-run list
    if slice_labels:
        slice_counts = write_slice_reports(data_list, output_dir, subdetectors=subdetectors)
        n_slice_flagged = sum(1 for d in data_list if d["primary_status"] == "GOOD" and d.get("slice_status", "GOOD") != "GOOD")
        print(f"\nCentrality slices (good runs only; FLAG_LOW_STATS not counted):")
        print(f"  Good runs with any slice flag: {n_slice_flagged:,}")
        for label in slice_labels:
            for sdet in subdetectors:
                c = slice_counts.get(label, {}).get(sdet, {})
                if c:
                    print(f"  {label:>6} {sdet:<2}: " + ", ".join(f"{k.replace('FLAG_', '')} {v}" for k, v in sorted(c.items())))
        slice_plot = output_dir / "slices_summary.png"
        plot_slice_summary(data_list, slice_plot, slice_labels=slice_labels, cuts=cuts)
        print(f"  - Slice summary plot saved: {slice_plot.name}")

    # 2. Determine which subdetectors to plot
    plots_subdets = ["NS"] if args.subdetector == "NS" else (["N"] if args.subdetector == "N" else (["S"] if args.subdetector == "S" else ["NS", "N", "S"]))

    # 3. Dedicated subfolder for Metric Distributions
    dist_dir = output_dir / "metric_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 4. Generate Aggregate Plots
    for sdet in plots_subdets:
        print(f"\nGenerating publication plots for subdetector: {sdet}...")
        max_a = cuts["max_a_ns"] if sdet == "NS" else cuts["max_a_arm"]

        # 2D Heatmap
        heatmap_path = output_dir / f"heatmap_psi2_{sdet}.png"
        plot_heatmap(data_list, heatmap_path, subdet=sdet, test_runs=args.test_runs)
        print(f"  - Heatmap saved: {heatmap_path.name}")

        # Trend plots vs run number
        trends_path = output_dir / f"trends_psi2_{sdet}.png"
        plot_trends(
            data_list,
            trends_path,
            subdet=sdet,
            test_runs=args.test_runs,
            rms_cut=cuts["max_rms_sigma"],
            max_chi2=cuts["max_chi2"],
            max_a=max_a,
            min_events=cuts["min_events"],
            save_individual=True,
        )
        print(f"  - Trend plots saved: {trends_path.name}")

        # Metric distribution histograms (saved in dedicated metric_distributions/ subfolder with individual plots)
        for path, individual in [(dist_dir / f"metric_distributions_4panel_{sdet}.png", True),
                                 (output_dir / f"metric_distributions_psi2_{sdet}.png", False)]:
            plot_metric_distributions(
                data_list,
                path,
                subdet=sdet,
                rms_cut=cuts["max_rms_sigma"],
                max_chi2=cuts["max_chi2"],
                max_a=max_a,
                min_events=cuts["min_events"],
                save_individual=individual,
            )
        print(f"  - Metric distributions saved in {dist_dir.name}/")

        # Ensemble Flatness Profile
        ens_path = output_dir / f"ensemble_profile_psi2_{sdet}.png"
        plot_ensemble_profile(data_list, ens_path, subdet=sdet)
        print(f"  - Ensemble profile saved: {ens_path.name}")

    # 5. Generate 1x3 side-by-side metric comparisons (S vs N vs NS) in metric_distributions/
    print(f"\nGenerating 1x3 side-by-side metric comparison plots (S vs N vs NS)...")
    for metric_type in ("rms", "chi2", "modulation"):
        for zoomed in (False, True):
            suffix = "_zoomed" if zoomed else ""
            plot_metric_comparison_1x3(
                data_list,
                dist_dir / f"comparison_1x3_{metric_type}{suffix}.png",
                metric_type=metric_type,
                rms_cut=cuts["max_rms_sigma"],
                max_chi2=cuts["max_chi2"],
                max_a=cuts["max_a_arm"],
                max_a_ns=cuts["max_a_ns"],
                zoomed=zoomed,
            )
    print(f"  - 1x3 Comparison plots saved in {dist_dir.name}/")

    # 6. Generate Q-vector component QA trends (recentering & flattening)
    has_comp = any("Q_N_x_max_abs" in d for d in data_list)
    if has_comp:
        print(f"\nGenerating aggregate Q-vector component QA trend plots...")
        plot_components_trends(
            data_list,
            output_dir,
            max_recent_dev=cuts["max_recent_dev"],
            max_twist_dev=cuts["max_twist_dev"],
            max_ratio_dev=cuts["max_ratio_dev"],
        )
        print(f"  - Component trend plots saved: trends_components_recentering.png, trends_components_flattening.png")

    # 7. Generate Top Flat Run Examples
    if args.top_flat_examples > 0:
        print(f"\nSelecting top {args.top_flat_examples} flattest runs and generating diagnostic reference plots...")
        top_runs = generate_top_flat_examples(
            data_list,
            output_dir,
            subdet="NS",
            n_examples=args.top_flat_examples,
        )
        print(f"  - Top flat runs saved: {top_runs}")

    # 8. Generate Flagged Arm Runs Diagnostics (dedicated folder for any runs with flagged detector arms)
    print(f"\nIdentifying runs with flagged detector arms and generating diagnostic plots...")
    flagged_arm_runs = generate_flagged_arm_examples(data_list, output_dir, cuts=cuts)
    print(f"  - Flagged arm runs saved in flagged_arm_runs/: {flagged_arm_runs}")

    # 9. Generate Failure Mode Galleries for primary subdetector (NS)
    print(f"\nSelecting representative failure mode examples and generating gallery...")
    examples_map = generate_failure_mode_examples(
        data_list,
        output_dir,
        subdet="NS",
        example_runs_per_mode=args.example_runs_per_mode,
        cuts=cuts,
    )

    for mode, runs in examples_map.items():
        if runs:
            print(f"  - {mode}: runs {runs}")

    # 10. Diagnostic plots for user-specified test runs
    if args.test_runs:
        test_dir = output_dir / "test_runs"
        test_dir.mkdir(parents=True, exist_ok=True)
        run_dict = {d["run_number"]: d for d in data_list}
        print(f"\nGenerating diagnostic plots for {len(args.test_runs)} test runs...")
        for tr in args.test_runs:
            if tr in run_dict:
                plot_diagnostic_run(run_dict[tr], test_dir / f"run_{tr}_diagnostic.png")
                print(f"  - Test run {tr} saved: run_{tr}_diagnostic.png")
            else:
                print(f"  - Warning: Test run {tr} not found in dataset.")

    print(f"\n============================================================")
    print(f"  QA processing complete! Outputs saved in: {output_dir}")
    print(f"============================================================")


if __name__ == "__main__":
    main()
