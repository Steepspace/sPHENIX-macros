#!/usr/bin/env python3
"""
plot_qvec_run.py

Standalone single-run or multi-run diagnostic inspector for sEPD Q-Vector flatness QA.
Takes one or more ROOT files or run numbers, computes flatness metrics and residual
Fourier harmonics, prints a summary table to the console, and generates a
high-resolution 3-arm diagnostic figure (NS, N, S) with Fourier fit curves.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure local imports from the same directory work reliably
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from qvec_qa_metrics import add_cut_args, cuts_from_args, extract_qvec_run_metrics
from qvec_qa_plots import plot_diagnostic_run, plot_qvec_components_diagnostic


def print_run_summary_table(metric_record: dict) -> None:
    """Print an aligned ASCII table of metrics for NS, N, and S arms."""
    run_number = metric_record["run_number"]
    status = metric_record.get("status", "UNKNOWN")
    print(f"\n==========================================================================================")
    print(f"  sEPD Q-Vector Run QA Diagnostic: Run {run_number} | Overall Status: {status}")
    print(f"  File: {metric_record['file_path']}")
    print(f"==========================================================================================")
    header = (f"{'Arm':<6} | {'Events':<10} | {'Chi2/ndf':<9} | {'RMS [sig]':<9} | {'Excess RMS':<11} | "
              f"{'MaxDev [sig]':<12} | {'A1 % (sig)':<12} | {'A2 % (sig)':<12} | {'Status'}")
    print(header)
    print("-" * len(header))

    subdets = ["NS", "N", "S"]
    for sdet in subdets:
        sm = metric_record.get("subdetectors", {}).get(sdet, {})
        evts = f"{sm.get('total_events', 0):,.0f}"
        chi2 = f"{sm.get('chi2_ndf', float('nan')):.3f}"
        rms = f"{sm.get('rms_sigma', float('nan')):.2f}"
        exc = f"{sm.get('rms_excess_pct', float('nan')):.2f}%"
        mdev = f"{sm.get('max_dev_sigma', float('nan')):.2f}"
        a1 = f"{sm.get('A1_pct', float('nan')):.2f} ({sm.get('A1_sig', float('nan')):.1f})"
        a2 = f"{sm.get('A2_pct', float('nan')):.2f} ({sm.get('A2_sig', float('nan')):.1f})"
        st = sm.get("status", "UNKNOWN")

        print(f"{sdet:<6} | {evts:<10} | {chi2:<9} | {rms:<9} | {exc:<11} | {mdev:<12} | {a1:<12} | {a2:<12} | {st}")
    print("------------------------------------------------------------------------------------------")

    # Intermediate Q-vector Component Calibration Checks
    comp_dict = metric_record.get("components")
    if comp_dict:
        print("  Q-Vector Intermediate Calibration Checks (Recentering & Flattening):")
        comp_sdet = comp_dict.get("subdetectors", {})
        for sdet in ["N", "S"]:
            cd = comp_sdet.get(sdet, {})
            if cd.get("has_recentering"):
                qx = cd.get("max_abs_qx", float("nan"))
                qy = cd.get("max_abs_qy", float("nan"))
                print(f"    {sdet} Arm Recentering:  max |<Qx>| = {qx:.3e}, max |<Qy>| = {qy:.3e}")
        for sdet in ["N", "S", "NS"]:
            cd = comp_sdet.get(sdet, {})
            qxy = cd.get("max_abs_qxy", float("nan"))
            r_dev = cd.get("max_dev_ratio_xxyy", float("nan"))
            print(f"    {sdet} Arm Flattening:   max |<Qxy>| = {qxy:.3e}, max |<Qxx>/<Qyy> - 1| = {r_dev:.3e}")
    print("==========================================================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect single or multiple sEPD Q-Vector calibration ROOT files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more ROOT files to inspect.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output plot path for diagnostic figure (default: run_<RUN>_qvec_diagnostic.png).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save diagnostic plots if processing multiple files (default: current dir).",
    )
    parser.add_argument(
        "--harmonic",
        type=int,
        default=2,
        help="Harmonic order n (default: 2).",
    )
    add_cut_args(parser)
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

    args = parser.parse_args()

    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            print(f"Error: File not found: {path}")
            continue

        res, err = extract_qvec_run_metrics(
            path,
            subdetectors=("NS", "N", "S"),
            harmonic=args.harmonic,
            pass_suffix="_corr2",
            cent_min=args.cent_min,
            cent_max=args.cent_max,
            cuts=cuts_from_args(args),
        )

        if err or res is None:
            print(f"Error reading {path.name}: {err}")
            continue

        print_run_summary_table(res)

        # Plot output path
        if args.output and len(args.files) == 1:
            out_plot = Path(args.output)
        else:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_plot = out_dir / f"run_{res['run_number']}_qvec_diagnostic.png"

        plot_diagnostic_run(res, out_plot)
        print(f"Diagnostic plot saved: {out_plot}")

        if res.get("components"):
            out_comp = out_plot.parent / f"run_{res['run_number']}_components_diagnostic.png"
            plot_qvec_components_diagnostic(res["run_number"], res["components"], out_comp)
            print(f"Component calibration plot saved: {out_comp}")


if __name__ == "__main__":
    main()
