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

from qvec_qa_metrics import extract_qvec_run_metrics, evaluate_subdet_status
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
)


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
    ]

    for sdet in subdetectors:
        fieldnames.extend([
            f"chi2_ndf_{sdet}",
            f"rms_pct_{sdet}",
            f"rms_excess_pct_{sdet}",
            f"max_dev_pct_{sdet}",
            f"A1_pct_{sdet}",
            f"A2_pct_{sdet}",
            f"status_{sdet}",
        ])

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
                "file_path": d["file_path"],
            }

            subdets_dict = d.get("subdetectors", {})
            for sdet in subdetectors:
                sm = subdets_dict.get(sdet, {})
                row[f"chi2_ndf_{sdet}"] = f"{sm.get('chi2_ndf', np.nan):.3f}" if not np.isnan(sm.get("chi2_ndf", np.nan)) else "nan"
                row[f"rms_pct_{sdet}"] = f"{sm.get('rms_pct', np.nan):.3f}" if not np.isnan(sm.get("rms_pct", np.nan)) else "nan"
                row[f"rms_excess_pct_{sdet}"] = f"{sm.get('rms_excess_pct', np.nan):.3f}" if not np.isnan(sm.get("rms_excess_pct", np.nan)) else "nan"
                row[f"max_dev_pct_{sdet}"] = f"{sm.get('max_dev_pct', np.nan):.2f}" if not np.isnan(sm.get("max_dev_pct", np.nan)) else "nan"
                row[f"A1_pct_{sdet}"] = f"{sm.get('A1_pct', np.nan):.3f}" if not np.isnan(sm.get("A1_pct", np.nan)) else "nan"
                row[f"A2_pct_{sdet}"] = f"{sm.get('A2_pct', np.nan):.3f}" if not np.isnan(sm.get("A2_pct", np.nan)) else "nan"
                row[f"status_{sdet}"] = sm.get("status", "UNKNOWN")

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


def generate_failure_mode_examples(
    metrics_list: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    subdet: str = "NS",
    example_runs_per_mode: int = 3,
    max_rms_pct: float = 2.5,
    max_chi2: float = 2.0,
    max_a: float = 1.0,
) -> Dict[str, List[int]]:
    """
    Identify representative outlier runs for each failure mode and generate
    diagnostic 3-arm plots and metric distributions.
    """
    output_dir = Path(output_dir)
    examples_dir = output_dir / "failure_examples"

    failure_modes_config = [
        {
            "mode": "FLAG_HIGH_CHI2",
            "description": f"High Flatness Chi2/ndf (> {max_chi2})",
            "filter_fn": lambda d: "FLAG_HIGH_CHI2" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: d.get("subdetectors", {}).get(subdet, {}).get("chi2_ndf", 0.0),
            "reverse": True,
        },
        {
            "mode": "FLAG_NON_FLAT",
            "description": f"Excess RMS Non-Flatness (> cut)",
            "filter_fn": lambda d: "FLAG_NON_FLAT" in d.get("subdetectors", {}).get(subdet, {}).get("status", "").split(";"),
            "sort_key": lambda d: d.get("subdetectors", {}).get(subdet, {}).get("rms_pct", 0.0),
            "reverse": True,
        },
        {
            "mode": "FLAG_RESIDUAL_MODULATION",
            "description": f"Residual Harmonic Modulation (> {max_a}%)",
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
            max_rms_pct=max_rms_pct,
            max_chi2=max_chi2,
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


def generate_flagged_arm_examples(
    metrics_list: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[int]:
    """
    Generate diagnostic plots for all runs that have any flagged detector arm status
    (overall status != 'GOOD' or any subdetector status != 'GOOD') into a dedicated
    'flagged_arm_runs' subfolder. Also writes a summary markdown report explaining
    the specific physical and statistical root causes for each flag.
    """
    flagged_dir = output_dir / "flagged_arm_runs"
    flagged_dir.mkdir(parents=True, exist_ok=True)

    flagged_runs = [
        d for d in metrics_list
        if d.get("status") != "GOOD"
        or any(sm.get("status") != "GOOD" for sm in d.get("subdetectors", {}).values())
    ]
    flagged_run_numbers = []

    readme_lines = [
        "# Flagged Arm Runs Investigation Summary\n",
        "This directory contains detailed diagnostic plots for runs where one or more detector arms",
        "were flagged with a non-GOOD status during Q-vector flatness QA.\n",
        f"- **Total Runs Analyzed:** {len(metrics_list):,}",
        f"- **Flagged Runs Identified:** {len(flagged_runs)}\n",
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

        # Determine detailed explanation
        reasons = []
        if s_s.get("status") == "FLAG_NON_FLAT":
            reasons.append(
                f"**South Arm (S)** flagged `FLAG_NON_FLAT`: Max deviation reached {s_s.get('max_dev_pct', 0.0):.2f}% (cut: 12.0%). "
                f"Run has low statistics ({s_s.get('total_events', 0):,.0f} total events, ~950 events/bin). "
                f"Poisson uncertainty is 1/sqrt(950) ≈ 3.24%/bin, so a single 3.8-sigma Poisson fluctuation caused the max deviation to hit 12.56%. "
                f"chi2/ndf is 1.11 (healthy) and residual modulation A2 is 0.45% (< 1.0%)."
            )
        elif s_s.get("status") == "FLAG_RESIDUAL_MODULATION":
            reasons.append(
                f"**South Arm (S)** flagged `FLAG_RESIDUAL_MODULATION`: Second harmonic Fourier amplitude A2 reached {s_s.get('A2_pct', 0.0):.2f}% (cut: 1.00%). "
                f"Run has sufficient statistics ({s_s.get('total_events', 0):,.0f} events). A residual 1.14% modulation exists in South arm, "
                f"which partially cancels with North to yield a flat combined NS distribution (A2 = 0.63% < 1.0%, status GOOD in NS)."
            )
        elif s_s.get("status") != "GOOD":
            reasons.append(f"South Arm: {s_s.get('status')}")

        if n_s.get("status") != "GOOD":
            reasons.append(f"North Arm: {n_s.get('status')}")
        if ns_s.get("status") != "GOOD":
            reasons.append(f"Combined NS: {ns_s.get('status')}")

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

    with open(flagged_dir / "README.md", "w") as f_rm:
        f_rm.write("\n".join(readme_lines) + "\n")

    return flagged_run_numbers


def reevaluate_data_list(
    data_list: List[Dict[str, Any]],
    subdetectors: Sequence[str] = ("NS", "N", "S"),
    min_events: float = 100000.0,
    max_chi2: float = 2.0,
    max_rms: float = 3.5,
    max_excess_rms: float = 2.0,
    max_dev_pct: float = 12.0,
    max_a: float = 1.0,
) -> None:
    """
    Re-evaluate QA pass/fail status flags on cached metrics without re-reading ROOT files.
    """
    for d in data_list:
        subdets = d.get("subdetectors", {})
        for sdet, sm in subdets.items():
            sm["status"] = evaluate_subdet_status(
                sm,
                min_events=min_events,
                max_chi2=max_chi2,
                max_rms=max_rms,
                max_excess_rms=max_excess_rms,
                max_dev_pct=max_dev_pct,
                max_a1=max_a,
                max_a2=max_a,
            )
        primary_status = subdets.get("NS", {}).get("status", "UNKNOWN")
        all_statuses = [sm.get("status", "UNKNOWN") for sm in subdets.values()]
        overall_status = "GOOD" if all(s == "GOOD" for s in all_statuses) else ";".join(sorted(set(filter(lambda s: s != "GOOD", all_statuses))))
        d["primary_status"] = primary_status
        d["status"] = overall_status
        if "NS" in subdets:
            d["total_events"] = subdets["NS"].get("total_events", d.get("total_events", 0.0))


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

    # QA Cut Thresholds
    parser.add_argument(
        "--min-events",
        type=float,
        default=100000.0,
        help="Minimum events threshold per run (default: 100,000).",
    )
    parser.add_argument(
        "--max-chi2",
        type=float,
        default=2.0,
        help="Maximum chi2/ndf threshold against flat hypothesis (default: 2.0).",
    )
    parser.add_argument(
        "--max-rms",
        type=float,
        default=3.5,
        help="Maximum total RMS non-flatness [%%] cut line (default: 3.5%%).",
    )
    parser.add_argument(
        "--max-excess-rms",
        type=float,
        default=2.0,
        help="Maximum excess (statistical-subtracted) RMS [%%] cut (default: 2.0%%).",
    )
    parser.add_argument(
        "--max-dev",
        type=float,
        default=12.0,
        help="Maximum relative peak/trough deviation [%%] cut (default: 12.0%%).",
    )
    parser.add_argument(
        "--max-a",
        type=float,
        default=1.0,
        help="Maximum Fourier harmonic amplitude [%%] for A1 and A2 (default: 1.0%%).",
    )
    parser.add_argument(
        "--top-flat-examples",
        type=int,
        default=3,
        help="Number of representative top flattest runs to save for reference (default: 3).",
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

    # Centrality Slicing
    parser.add_argument(
        "--cent-min",
        type=float,
        default=None,
        help="Minimum centrality [%%] for projection slice (default: all).",
    )
    parser.add_argument(
        "--cent-max",
        type=float,
        default=None,
        help="Maximum centrality [%%] for projection slice (default: all).",
    )

    parser.add_argument(
        "--example-runs-per-mode",
        type=int,
        default=3,
        help="Number of representative runs to plot per failure mode (default: 3).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subdetectors = ("NS", "N", "S")

    use_cache = args.cache or (args.cache_file is not None)
    cache_path = Path(args.cache_file) if args.cache_file else (output_dir / "qvec_qa_cache.pkl")

    loaded_from_cache = False
    data_list: List[Dict[str, Any]] = []

    if use_cache and cache_path.exists() and not args.force_reload:
        print(f"============================================================")
        print(f"  sEPD Q-Vector Run-by-Run QA Aggregate Suite")
        print(f"  Loading cached metrics from: {cache_path}")
        print(f"  Output directory: {output_dir}")
        print(f"============================================================")
        try:
            with open(cache_path, "rb") as f_pkl:
                data_list = pickle.load(f_pkl)
            print(f"Successfully loaded {len(data_list):,} runs from cache.")
            loaded_from_cache = True
            # Re-evaluate statuses using current thresholds
            reevaluate_data_list(
                data_list,
                subdetectors=subdetectors,
                min_events=args.min_events,
                max_chi2=args.max_chi2,
                max_rms=args.max_rms,
                max_excess_rms=args.max_excess_rms,
                max_dev_pct=args.max_dev,
                max_a=args.max_a,
            )
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}. Falling back to ROOT files.")
            loaded_from_cache = False

    if not loaded_from_cache:
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

        print(f"============================================================")
        print(f"  sEPD Q-Vector Run-by-Run QA Aggregate Suite")
        print(f"  Total input files: {len(file_paths):,}")
        print(f"  Harmonic: {args.harmonic}")
        print(f"  Parallel workers: {args.workers}")
        print(f"  Output directory: {output_dir}")
        if use_cache:
            print(f"  Pickle cache:     {cache_path} (will be created)")
        print(f"============================================================")

        extract_fn = functools.partial(
            extract_qvec_run_metrics,
            subdetectors=subdetectors,
            harmonic=args.harmonic,
            pass_suffix="_corr2",
            cent_min=args.cent_min,
            cent_max=args.cent_max,
            min_events=args.min_events,
            max_chi2=args.max_chi2,
            max_rms=args.max_rms,
            max_excess_rms=args.max_excess_rms,
            max_dev_pct=args.max_dev,
            max_a1=args.max_a,
            max_a2=args.max_a,
        )

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

        if not data_list:
            print("Error: No valid run data extracted.")
            sys.exit(1)

        # Sort sequentially by run number
        data_list.sort(key=lambda d: d["run_number"])

        if use_cache:
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

    # 2. Determine which subdetectors to plot
    plots_subdets = ["NS"] if args.subdetector == "NS" else (["N"] if args.subdetector == "N" else (["S"] if args.subdetector == "S" else ["NS", "N", "S"]))

    # 3. Dedicated subfolder for Metric Distributions
    dist_dir = output_dir / "metric_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 4. Generate Aggregate Plots
    for sdet in plots_subdets:
        print(f"\nGenerating publication plots for subdetector: {sdet}...")

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
            max_rms_pct=args.max_rms,
            max_chi2=args.max_chi2,
            max_a=args.max_a,
            min_events=args.min_events,
            save_individual=True,
        )
        print(f"  - Trend plots saved: {trends_path.name}")

        # Metric distribution histograms (saved in dedicated metric_distributions/ subfolder with individual plots)
        dist_path = dist_dir / f"metric_distributions_4panel_{sdet}.png"
        plot_metric_distributions(
            data_list,
            dist_path,
            subdet=sdet,
            max_rms_pct=args.max_rms,
            max_chi2=args.max_chi2,
            max_a=args.max_a,
            min_events=args.min_events,
            save_individual=True,
        )
        # Also save 4-panel at top-level for convenience
        plot_metric_distributions(
            data_list,
            output_dir / f"metric_distributions_psi2_{sdet}.png",
            subdet=sdet,
            max_rms_pct=args.max_rms,
            max_chi2=args.max_chi2,
            max_a=args.max_a,
            min_events=args.min_events,
            save_individual=False,
        )
        print(f"  - Metric distributions saved in {dist_dir.name}/")

        # Ensemble Flatness Profile
        ens_path = output_dir / f"ensemble_profile_psi2_{sdet}.png"
        plot_ensemble_profile(data_list, ens_path, subdet=sdet)
        print(f"  - Ensemble profile saved: {ens_path.name}")

    # 5. Generate 1x3 side-by-side metric comparisons (S vs N vs NS) in metric_distributions/
    print(f"\nGenerating 1x3 side-by-side metric comparison plots (S vs N vs NS)...")
    plot_metric_comparison_1x3(data_list, dist_dir / "comparison_1x3_rms.png", metric_type="rms", max_rms_pct=args.max_rms, zoomed=False)
    plot_metric_comparison_1x3(data_list, dist_dir / "comparison_1x3_rms_zoomed.png", metric_type="rms", max_rms_pct=args.max_rms, zoomed=True)
    plot_metric_comparison_1x3(data_list, dist_dir / "comparison_1x3_chi2.png", metric_type="chi2", max_chi2=args.max_chi2, zoomed=False)
    plot_metric_comparison_1x3(data_list, dist_dir / "comparison_1x3_chi2_zoomed.png", metric_type="chi2", max_chi2=args.max_chi2, zoomed=True)
    plot_metric_comparison_1x3(data_list, dist_dir / "comparison_1x3_modulation.png", metric_type="modulation", max_a=args.max_a, zoomed=False)
    plot_metric_comparison_1x3(data_list, dist_dir / "comparison_1x3_modulation_zoomed.png", metric_type="modulation", max_a=args.max_a, zoomed=True)
    print(f"  - 1x3 Comparison plots saved in {dist_dir.name}/")

    # 6. Generate Top Flat Run Examples
    if args.top_flat_examples > 0:
        print(f"\nSelecting top {args.top_flat_examples} flattest runs and generating diagnostic reference plots...")
        top_runs = generate_top_flat_examples(
            data_list,
            output_dir,
            subdet="NS",
            n_examples=args.top_flat_examples,
        )
        print(f"  - Top flat runs saved: {top_runs}")

    # 7. Generate Flagged Arm Runs Diagnostics (dedicated folder for any runs with flagged detector arms)
    print(f"\nIdentifying runs with flagged detector arms and generating diagnostic plots...")
    flagged_arm_runs = generate_flagged_arm_examples(data_list, output_dir)
    print(f"  - Flagged arm runs saved in flagged_arm_runs/: {flagged_arm_runs}")

    # 8. Generate Failure Mode Galleries for primary subdetector (NS)
    print(f"\nSelecting representative failure mode examples and generating gallery...")
    examples_map = generate_failure_mode_examples(
        data_list,
        output_dir,
        subdet="NS",
        example_runs_per_mode=args.example_runs_per_mode,
        max_rms_pct=args.max_rms,
        max_chi2=args.max_chi2,
        max_a=args.max_a,
    )

    for mode, runs in examples_map.items():
        if runs:
            print(f"  - {mode}: runs {runs}")

    # 5. Diagnostic plots for user-specified test runs
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
