#!/usr/bin/env python3
"""
qvec_qa_metrics.py

Fast extraction of scalar flatness and QA metrics from sEPD Q-Vector ROOT files.
Extracts y-projections of 2D event-plane histograms (e.g. h2_sEPD_Psi_{subdet}_2_corr2)
and evaluates chi2/ndf, RMS non-flatness, excess RMS, maximum deviation, and
Fourier harmonic modulations (A1 dipole, A2 quadrupole).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import uproot


def parse_run_number(filename_or_path: Union[str, Path]) -> Optional[int]:
    """
    Extract the integer run number from a file name or path.
    Examples:
        QVecCalib-67597.root -> 67597
        QA-67597.root        -> 67597
        67597.root           -> 67597
    """
    name = Path(filename_or_path).name
    # Match pattern like QVecCalib-12345.root or QA-12345.root
    match = re.search(r"[-_](\d+)\.root$", name)
    if match:
        return int(match.group(1))

    # Match pure run number like 12345.root
    match = re.match(r"^(\d+)\.root$", name)
    if match:
        return int(match.group(1))

    # Generic digit match
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())

    return None


def get_qvec_psi_hist_data(
    f: uproot.ReadOnlyDirectory,
    hist_name: str,
    cent_min: Optional[float] = None,
    cent_max: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Retrieve 1D 2Psi2 histogram data from a 2D (centrality vs 2Psi2) histogram in an open uproot file.
    Projects onto the y-axis (2Psi2).

    Parameters:
        f: Open uproot directory/file.
        hist_name: Name of the 2D histogram (e.g. 'h2_sEPD_Psi_NS_2_corr2').
        cent_min: Optional minimum centrality [%] for projection slice.
        cent_max: Optional maximum centrality [%] for projection slice.

    Returns:
        tuple: (values_1d, edges_y, bin_centers_y, total_events)
               or (None, None, None, 0.0) if histogram not found.
    """
    if hist_name not in f:
        return None, None, None, 0.0

    item = f[hist_name]
    arr = item.to_numpy()

    # Case 1: 2D histogram (values_2d, xedges, yedges)
    if len(arr) == 3:
        values_2d, edges_x, edges_y = arr
        x_centers = 0.5 * (edges_x[:-1] + edges_x[1:])

        # Apply centrality slice if specified
        if cent_min is not None or cent_max is not None:
            c_low = cent_min if cent_min is not None else -np.inf
            c_high = cent_max if cent_max is not None else np.inf
            cent_mask = (x_centers >= c_low) & (x_centers <= c_high)
            if np.any(cent_mask):
                values_1d = np.sum(values_2d[cent_mask, :], axis=0)
            else:
                values_1d = np.zeros(len(edges_y) - 1)
        else:
            # Integrate over all centrality bins
            values_1d = np.sum(values_2d, axis=0)

        bin_centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
        total_events = float(np.sum(values_1d))
        return values_1d, edges_y, bin_centers_y, total_events

    # Case 2: Already 1D histogram (values_1d, edges)
    elif len(arr) == 2:
        values_1d, edges_y = arr
        bin_centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
        total_events = float(np.sum(values_1d))
        return values_1d, edges_y, bin_centers_y, total_events

    return None, None, None, 0.0


def compute_psi_flatness_metrics(
    values: np.ndarray,
    edges: np.ndarray,
    syst_floor: float = 0.01,
) -> Dict[str, Any]:
    """
    Compute scalar flatness and harmonic QA metrics for a 1D event plane distribution.

    Parameters:
        values: 1D array of bin counts across 2Psi2 in [-pi, pi].
        edges: 1D array of bin edges.
        syst_floor: Relative systematic floor for chi2 calculation (default: 0.01 = 1%).

    Returns:
        dict containing flatness and harmonic metrics.
    """
    n_bins = len(values)
    total_events = float(np.sum(values))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    if total_events <= 0 or n_bins < 2:
        return {
            "total_events": 0.0,
            "mean_bin": 0.0,
            "rms_pct": np.nan,
            "rms_excess_pct": np.nan,
            "max_dev_pct": np.nan,
            "chi2_ndf": np.nan,
            "a1": np.nan,
            "b1": np.nan,
            "A1_pct": np.nan,
            "a2": np.nan,
            "b2": np.nan,
            "A2_pct": np.nan,
            "values": values,
            "ratios": np.zeros(n_bins),
            "bin_centers": bin_centers,
            "edges": edges,
        }

    mean_bin = total_events / n_bins
    ratios = values / mean_bin
    dev = ratios - 1.0

    # 1. Raw RMS relative variation (%)
    tot_var = float(np.mean(dev ** 2))
    rms_pct = float(np.sqrt(tot_var)) * 100.0

    # 2. Excess (statistical-subtracted) RMS (%)
    # Expected Poisson variance on ratio is 1 / mean_bin
    stat_var = 1.0 / mean_bin
    excess_var = max(0.0, tot_var - stat_var)
    rms_excess_pct = float(np.sqrt(excess_var)) * 100.0

    # 3. Peak-to-peak maximum relative deviation (%)
    max_dev_pct = float(np.max(np.abs(dev))) * 100.0

    # 4. Chi2 / ndf against flat mean with Poisson variance + systematic floor
    var_i = np.where(values > 0, values, 1.0) + (syst_floor * mean_bin) ** 2
    chi2 = float(np.sum(((values - mean_bin) ** 2) / var_i))
    ndf = max(1, n_bins - 1)
    chi2_ndf = chi2 / ndf

    # 5. Fourier harmonic modulations:
    # 2Psi2 is defined in [-pi, pi].
    # Residual first harmonic (dipole offset from recentering):
    # a1 = 2 <cos(phi)>, b1 = 2 <sin(phi)>, A1 = sqrt(a1^2 + b1^2)
    a1 = float(2.0 * np.sum(values * np.cos(bin_centers)) / total_events)
    b1 = float(2.0 * np.sum(values * np.sin(bin_centers)) / total_events)
    A1_pct = float(np.sqrt(a1 ** 2 + b1 ** 2)) * 100.0

    # Residual second harmonic (quadrupole offset from flattening):
    # a2 = 2 <cos(2 phi)>, b2 = 2 <sin(2 phi)>, A2 = sqrt(a2^2 + b2^2)
    a2 = float(2.0 * np.sum(values * np.cos(2.0 * bin_centers)) / total_events)
    b2 = float(2.0 * np.sum(values * np.sin(2.0 * bin_centers)) / total_events)
    A2_pct = float(np.sqrt(a2 ** 2 + b2 ** 2)) * 100.0

    return {
        "total_events": total_events,
        "mean_bin": mean_bin,
        "rms_pct": rms_pct,
        "rms_excess_pct": rms_excess_pct,
        "max_dev_pct": max_dev_pct,
        "chi2_ndf": chi2_ndf,
        "a1": a1,
        "b1": b1,
        "A1_pct": A1_pct,
        "a2": a2,
        "b2": b2,
        "A2_pct": A2_pct,
        "values": values,
        "ratios": ratios,
        "bin_centers": bin_centers,
        "edges": edges,
    }


def evaluate_subdet_status(
    m: Dict[str, Any],
    min_events: float = 100000.0,
    max_chi2: float = 2.0,
    max_rms: float = 3.5,
    max_excess_rms: float = 2.0,
    max_dev_pct: float = 12.0,
    max_a1: float = 1.0,
    max_a2: float = 1.0,
) -> str:
    """
    Evaluate QA pass/fail status flags for a single subdetector metric dictionary.
    """
    if m["total_events"] <= 0 or np.isnan(m["rms_pct"]):
        return "FLAG_EMPTY_OR_ZERO"

    flags = []
    if m["total_events"] < min_events:
        flags.append("FLAG_LOW_STATS")

    if not np.isnan(m["chi2_ndf"]) and m["chi2_ndf"] > max_chi2:
        flags.append("FLAG_HIGH_CHI2")

    if (not np.isnan(m["rms_excess_pct"]) and m["rms_excess_pct"] > max_excess_rms) or \
       (not np.isnan(m["rms_pct"]) and m["rms_pct"] > max_rms and not np.isnan(m["chi2_ndf"]) and m["chi2_ndf"] > max_chi2) or \
       (not np.isnan(m["max_dev_pct"]) and m["max_dev_pct"] > max_dev_pct):
        flags.append("FLAG_NON_FLAT")

    if (not np.isnan(m["A1_pct"]) and m["A1_pct"] > max_a1) or \
       (not np.isnan(m["A2_pct"]) and m["A2_pct"] > max_a2):
        flags.append("FLAG_RESIDUAL_MODULATION")

    return ";".join(flags) if flags else "GOOD"


def extract_qvec_run_metrics(
    file_path: Union[str, Path],
    subdetectors: Tuple[str, ...] = ("NS", "N", "S"),
    harmonic: int = 2,
    pass_suffix: str = "_corr2",
    cent_min: Optional[float] = None,
    cent_max: Optional[float] = None,
    min_events: float = 100000.0,
    max_chi2: float = 2.0,
    max_rms: float = 3.5,
    max_excess_rms: float = 2.0,
    max_dev_pct: float = 12.0,
    max_a1: float = 1.0,
    max_a2: float = 1.0,
    syst_floor: float = 0.01,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fast extraction of scalar flatness and QA metrics from a single ROOT file
    for all specified subdetectors (default: NS, N, S).

    Returns:
        tuple: (metric_dict, error_message)
    """
    path = Path(file_path)
    if not path.exists():
        return None, f"File not found: {path}"

    run_number = parse_run_number(path)
    if run_number is None:
        return None, f"Could not parse run number from {path.name}"

    try:
        subdet_metrics = {}
        subdet_statuses = {}

        with uproot.open(path) as f_root:
            for subdet in subdetectors:
                hist_name = f"h2_sEPD_Psi_{subdet}_{harmonic}{pass_suffix}"
                values, edges, bin_centers, total_events = get_qvec_psi_hist_data(
                    f_root,
                    hist_name=hist_name,
                    cent_min=cent_min,
                    cent_max=cent_max,
                )

                if values is None:
                    # Histogram missing
                    m = compute_psi_flatness_metrics(np.array([]), np.array([]))
                    m["hist_name"] = hist_name
                    status = "FLAG_EMPTY_OR_ZERO"
                else:
                    m = compute_psi_flatness_metrics(values, edges, syst_floor=syst_floor)
                    m["hist_name"] = hist_name
                    status = evaluate_subdet_status(
                        m,
                        min_events=min_events,
                        max_chi2=max_chi2,
                        max_rms=max_rms,
                        max_excess_rms=max_excess_rms,
                        max_dev_pct=max_dev_pct,
                        max_a1=max_a1,
                        max_a2=max_a2,
                    )

                m["status"] = status
                subdet_metrics[subdet] = m
                subdet_statuses[subdet] = status

        # Overall composite status:
        # If NS is in subdetectors, primary status reflects NS, with arm-specific notes if N or S fail
        if "NS" in subdet_statuses:
            primary_status = subdet_statuses["NS"]
        else:
            # First available subdetector
            primary_status = list(subdet_statuses.values())[0]

        all_flags = []
        for sdet, st in subdet_statuses.items():
            if st != "GOOD":
                for flag in st.split(";"):
                    entry = f"{flag}_{sdet}" if len(subdetectors) > 1 else flag
                    if entry not in all_flags:
                        all_flags.append(entry)

        overall_status = ";".join(all_flags) if all_flags else "GOOD"

        # Primary total events (from NS if present, else first subdet)
        primary_tot_events = subdet_metrics.get("NS", {}).get("total_events", 0.0)

        record: Dict[str, Any] = {
            "run_number": run_number,
            "file_path": str(path),
            "harmonic": harmonic,
            "pass_suffix": pass_suffix,
            "total_events": primary_tot_events,
            "status": overall_status,
            "primary_status": primary_status,
            "subdetectors": subdet_metrics,
        }

        # Flatten primary subdetector (NS) fields at top-level for convenience
        if "NS" in subdet_metrics:
            ns_m = subdet_metrics["NS"]
            record.update({
                "mean_bin": ns_m["mean_bin"],
                "rms_pct": ns_m["rms_pct"],
                "rms_excess_pct": ns_m["rms_excess_pct"],
                "max_dev_pct": ns_m["max_dev_pct"],
                "chi2_ndf": ns_m["chi2_ndf"],
                "A1_pct": ns_m["A1_pct"],
                "A2_pct": ns_m["A2_pct"],
                "values": ns_m["values"],
                "ratios": ns_m["ratios"],
                "bin_centers": ns_m["bin_centers"],
                "edges": ns_m["edges"],
            })

        return record, None

    except Exception as e:
        return None, f"Error processing {path.name}: {e}"
