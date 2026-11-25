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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
            "rms_sigma": np.nan,
            "rms_excess_pct": np.nan,
            "max_dev_pct": np.nan,
            "max_dev_sigma": np.nan,
            "chi2_ndf": np.nan,
            "a1": np.nan,
            "b1": np.nan,
            "A1_pct": np.nan,
            "A1_sig": np.nan,
            "a2": np.nan,
            "b2": np.nan,
            "A2_pct": np.nan,
            "A2_sig": np.nan,
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

    # 6. Statistics-aware versions, in units of the Poisson expectation for a flat distribution:
    # - rms_sigma: RMS relative deviation / (1/sqrt(mean_bin)); ~1 for a statistically flat distribution
    # - max_dev_sigma: largest single-bin deviation in Poisson sigma of that bin
    # - A_n_sig: A_n / sigma(A_n), with sigma(a_n) = sigma(b_n) = sqrt(2/N) for a flat distribution
    rms_sigma = float(np.sqrt(tot_var * mean_bin))
    max_dev_sigma = float(np.max(np.abs(dev)) * np.sqrt(mean_bin))
    sigma_a = float(np.sqrt(2.0 / total_events)) * 100.0
    A1_sig = A1_pct / sigma_a
    A2_sig = A2_pct / sigma_a

    return {
        "total_events": total_events,
        "mean_bin": mean_bin,
        "rms_pct": rms_pct,
        "rms_sigma": rms_sigma,
        "rms_excess_pct": rms_excess_pct,
        "max_dev_pct": max_dev_pct,
        "max_dev_sigma": max_dev_sigma,
        "chi2_ndf": chi2_ndf,
        "a1": a1,
        "b1": b1,
        "A1_pct": A1_pct,
        "A1_sig": A1_sig,
        "a2": a2,
        "b2": b2,
        "A2_pct": A2_pct,
        "A2_sig": A2_sig,
        "values": values,
        "ratios": ratios,
        "bin_centers": bin_centers,
        "edges": edges,
    }


# ------------------------------------------------------------------------------
# QA cuts
# ------------------------------------------------------------------------------

# Default cuts. Statistics-dependent quantities are judged in Poisson sigma, and a flag needs
# both statistical significance and a size that matters, so low-statistics runs are not flagged
# for fluctuations and high-statistics runs are not flagged for tiny, real residuals.
DEFAULT_CUTS: Dict[str, float] = {
    "min_events": 100000.0,  # events per run (LOW_STATS)
    "max_chi2": 2.0,         # chi2/ndf vs flat, with 1% systematic floor per bin (HIGH_CHI2)
    "max_rms_sigma": 1.3,    # RMS deviation / Poisson expectation; NON_FLAT needs this AND max_excess_rms
    "max_excess_rms": 2.0,   # statistics-subtracted RMS [%]
    "max_dev_sigma": 6.0,    # largest single-bin deviation [Poisson sigma] (NON_FLAT)
    "max_a_ns": 0.75,        # A1/A2 [%] for NS; RESIDUAL_MODULATION needs this AND min_a_sig
    "max_a_arm": 1.0,        # A1/A2 [%] for N and S
    "min_a_sig": 4.0,        # A1/A2 significance [sigma of a flat distribution]
    "max_recent_dev": 1e-3,  # max |<Qx>|, |<Qy>| after recentering (N, S)
    "max_twist_dev": 1e-3,   # max |<Qxy>| after flattening (N, S, NS)
    "max_ratio_dev": 0.02,   # max |<Qxx>/<Qyy> - 1| after flattening (N, S, NS)
}

# Default centrality slices [%] evaluated in addition to the integrated distribution.
DEFAULT_CENT_SLICES: Tuple[Tuple[float, float], ...] = ((0, 20), (20, 40), (40, 60), (60, 80))


def add_cut_args(parser) -> None:
    """Adds the QA cut options (defaults from DEFAULT_CUTS) to an argparse parser."""
    c = DEFAULT_CUTS
    parser.add_argument("--min-events", type=float, default=c["min_events"],
                        help=f"Minimum events per run (default: {c['min_events']:,.0f}).")
    parser.add_argument("--max-chi2", type=float, default=c["max_chi2"],
                        help=f"Maximum chi2/ndf vs flat, with a 1%% systematic floor per bin (default: {c['max_chi2']:g}).")
    parser.add_argument("--max-rms-sigma", type=float, default=c["max_rms_sigma"],
                        help=f"RMS deviation in units of the Poisson expectation above which a run can be NON_FLAT; "
                             f"it must also exceed --max-excess-rms (default: {c['max_rms_sigma']:g}).")
    parser.add_argument("--max-excess-rms", type=float, default=c["max_excess_rms"],
                        help=f"Statistics-subtracted RMS [%%] above which a run can be NON_FLAT; "
                             f"it must also exceed --max-rms-sigma (default: {c['max_excess_rms']:g}%%).")
    parser.add_argument("--max-dev-sigma", type=float, default=c["max_dev_sigma"],
                        help=f"Largest single-bin deviation [Poisson sigma] for NON_FLAT (default: {c['max_dev_sigma']:g}).")
    parser.add_argument("--max-a", dest="max_a_ns", type=float, default=c["max_a_ns"],
                        help=f"Maximum A1/A2 [%%] for the combined NS distribution (default: {c['max_a_ns']:g}%%).")
    parser.add_argument("--max-a-arm", type=float, default=c["max_a_arm"],
                        help=f"Maximum A1/A2 [%%] for the N and S arms (default: {c['max_a_arm']:g}%%).")
    parser.add_argument("--min-a-sig", type=float, default=c["min_a_sig"],
                        help=f"Minimum A1/A2 significance [sigma] for RESIDUAL_MODULATION (default: {c['min_a_sig']:g}).")
    parser.add_argument("--max-recent-dev", type=float, default=c["max_recent_dev"],
                        help=f"Maximum |<Qx>|, |<Qy>| after recentering (default: {c['max_recent_dev']:g}).")
    parser.add_argument("--max-twist-dev", type=float, default=c["max_twist_dev"],
                        help=f"Maximum |<Qxy>| after flattening (default: {c['max_twist_dev']:g}).")
    parser.add_argument("--max-ratio-dev", type=float, default=c["max_ratio_dev"],
                        help=f"Maximum |<Qxx>/<Qyy> - 1| after flattening (default: {c['max_ratio_dev']:g}).")


def cuts_from_args(args) -> Dict[str, float]:
    """Builds the cuts dictionary from parsed add_cut_args() options."""
    return {k: getattr(args, k) for k in DEFAULT_CUTS}


def parse_cent_slices(text: str) -> Tuple[Tuple[float, float], ...]:
    """Parses '0-20,20-40' into ((0, 20), (20, 40)); 'none' or '' gives no slices."""
    if not text or text.strip().lower() == "none":
        return ()
    slices = []
    for part in text.split(","):
        lo, hi = part.strip().split("-")
        slices.append((float(lo), float(hi)))
    return tuple(slices)


def slice_label(cent_slice: Tuple[float, float]) -> str:
    return f"{cent_slice[0]:g}-{cent_slice[1]:g}"


def ensure_derived_metrics(m: Dict[str, Any]) -> None:
    """Adds metrics introduced later (e.g. rms_sigma) to records from older caches, using the stored histogram."""
    if "rms_sigma" in m:
        return
    values, edges = m.get("values"), m.get("edges")
    if values is not None and edges is not None and len(values) > 1:
        m.update({k: v for k, v in compute_psi_flatness_metrics(values, edges).items()
                  if k in ("rms_sigma", "max_dev_sigma", "A1_sig", "A2_sig")})
    else:
        m.update({"rms_sigma": np.nan, "max_dev_sigma": np.nan, "A1_sig": np.nan, "A2_sig": np.nan})


def _gt(value: float, cut: float) -> bool:
    return value is not None and not np.isnan(value) and value > cut


def evaluate_subdet_status(m: Dict[str, Any], cuts: Dict[str, float], subdet: str = "NS") -> str:
    """
    Evaluates the QA flags of one subdetector's flatness metrics:
    - FLAG_EMPTY_OR_ZERO: no events
    - FLAG_LOW_STATS: fewer than min_events
    - FLAG_HIGH_CHI2: chi2/ndf vs flat above max_chi2
    - FLAG_NON_FLAT: RMS both significant (rms_sigma > max_rms_sigma) and large (excess RMS > max_excess_rms),
      or a single bin more than max_dev_sigma Poisson sigma from flat
    - FLAG_RESIDUAL_MODULATION: A1 or A2 above the cut (max_a_ns for NS, max_a_arm for N/S)
      and more significant than min_a_sig
    """
    if m.get("total_events", 0) <= 0 or np.isnan(m.get("rms_pct", np.nan)):
        return "FLAG_EMPTY_OR_ZERO"
    ensure_derived_metrics(m)

    flags = []
    if m["total_events"] < cuts["min_events"]:
        flags.append("FLAG_LOW_STATS")

    if _gt(m["chi2_ndf"], cuts["max_chi2"]):
        flags.append("FLAG_HIGH_CHI2")

    rms_flag = _gt(m["rms_sigma"], cuts["max_rms_sigma"]) and _gt(m["rms_excess_pct"], cuts["max_excess_rms"])
    if rms_flag or _gt(m["max_dev_sigma"], cuts["max_dev_sigma"]):
        flags.append("FLAG_NON_FLAT")

    max_a = cuts["max_a_ns"] if subdet == "NS" else cuts["max_a_arm"]
    if any(_gt(m[f"A{n}_pct"], max_a) and _gt(m[f"A{n}_sig"], cuts["min_a_sig"]) for n in (1, 2)):
        flags.append("FLAG_RESIDUAL_MODULATION")

    return ";".join(flags) if flags else "GOOD"


def _component_flags(record: Dict[str, Any], cuts: Dict[str, float]) -> List[str]:
    """Recentering / twist / variance-ratio flags from the component scalars of a run record."""
    flags = []
    for sdet in ("N", "S"):
        if _gt(record.get(f"Q_{sdet}_x_max_abs", np.nan), cuts["max_recent_dev"]) or \
           _gt(record.get(f"Q_{sdet}_y_max_abs", np.nan), cuts["max_recent_dev"]):
            flags.append(f"FLAG_RECENT_NONZERO_{sdet}")
    for sdet in ("N", "S", "NS"):
        if _gt(record.get(f"Q_{sdet}_xy_max_abs", np.nan), cuts["max_twist_dev"]):
            flags.append(f"FLAG_TWIST_NONZERO_{sdet}")
        if _gt(record.get(f"Q_{sdet}_xxyy_max_dev", np.nan), cuts["max_ratio_dev"]):
            flags.append(f"FLAG_FLAT_ECCENTRICITY_{sdet}")
    return flags


def evaluate_run(record: Dict[str, Any], cuts: Dict[str, float]) -> None:
    """
    Sets all statuses of a run record from its metrics:
    - per subdetector (integrated over centrality): sm["status"]
    - record["primary_status"]: the NS status; decides good / flagged runs
    - record["status"]: all integrated and component flags, as FLAG_<name>_<subdet>
    - per centrality slice: record["slices"][label][subdet]["status"], and record["slice_status"] listing
      slice flags as FLAG_<name>_<subdet>@<slice>. Slice flags are reported only and do not change the
      primary status. FLAG_LOW_STATS is not listed for slices, since a slice holds only part of a run's
      events; the other flags are statistics-aware and are listed as usual.
    """
    subdets = record.get("subdetectors", {})
    all_flags: List[str] = []
    for sdet, sm in subdets.items():
        sm["status"] = evaluate_subdet_status(sm, cuts, sdet)
        if sm["status"] != "GOOD":
            all_flags.extend(f"{f}_{sdet}" for f in sm["status"].split(";") if f"{f}_{sdet}" not in all_flags)
    all_flags.extend(f for f in _component_flags(record, cuts) if f not in all_flags)

    record["primary_status"] = subdets["NS"]["status"] if "NS" in subdets else next(iter(subdets.values()), {}).get("status", "UNKNOWN")
    record["status"] = ";".join(all_flags) if all_flags else "GOOD"
    if "NS" in subdets:
        record["total_events"] = subdets["NS"].get("total_events", record.get("total_events", 0.0))

    slice_flags: List[str] = []
    for label, slice_subdets in record.get("slices", {}).items():
        for sdet, sm in slice_subdets.items():
            sm["status"] = evaluate_subdet_status(sm, cuts, sdet)
            slice_flags.extend(f"{f}_{sdet}@{label}" for f in sm["status"].split(";")
                               if f not in ("GOOD", "FLAG_LOW_STATS", "FLAG_EMPTY_OR_ZERO"))
    record["slice_status"] = ";".join(slice_flags) if slice_flags else "GOOD"

def extract_qvec_component_metrics(
    f: uproot.ReadOnlyDirectory,
    harmonic: int = 2,
    subdetectors: Sequence[str] = ("NS", "N", "S"),
    max_recent_dev: float = 1e-3,
    max_twist_dev: float = 1e-3,
    max_ratio_dev: float = 0.02,
) -> Dict[str, Any]:
    """
    Extract intermediate Q-vector calibration QA metrics across all centralities:
      1. Recentering checks: <Qx> and <Qy> are zero for all centrality bins (N and S)
         h_sEPD_Q_{N,S}_x_{harmonic}_corr2_avg, h_sEPD_Q_{N,S}_y_{harmonic}_corr2_avg
      2. Flattening twisting checks: <Qxy> is zero for all centrality bins (N, S, NS)
         h_sEPD_Q_{N,S,NS}_xy_{harmonic}_corr_avg
      3. Flattening variance checks: <Qxx>/<Qyy> is 1.0 for all centrality bins (N, S, NS)
         h_sEPD_Q_{N,S,NS}_xx_{harmonic}_corr_avg / h_sEPD_Q_{N,S,NS}_yy_{harmonic}_corr_avg
    """
    comp_dict: Dict[str, Any] = {
        "subdetectors": {},
        "flags": [],
    }
    flat_scalars: Dict[str, float] = {}

    for sdet in subdetectors:
        sdet_data: Dict[str, Any] = {
            "has_recentering": False,
            "has_twisting": False,
            "has_flattening_ratio": False,
        }

        # 1. Recentering (N and S)
        if sdet in ["N", "S"]:
            px_name = f"h_sEPD_Q_{sdet}_x_{harmonic}_corr2_avg"
            py_name = f"h_sEPD_Q_{sdet}_y_{harmonic}_corr2_avg"
            px = f.get(px_name)
            py = f.get(py_name)

            if px is not None and py is not None:
                sdet_data["has_recentering"] = True
                edges = px.axis().edges()
                cent_centers = 0.5 * (edges[:-1] + edges[1:])
                vx = px.values(flow=False)
                vy = py.values(flow=False)
                cx = px.counts(flow=False)
                cy = py.counts(flow=False)
                valid = (cx > 0) & (cy > 0)

                if np.any(valid):
                    max_abs_qx = float(np.max(np.abs(vx[valid])))
                    mean_abs_qx = float(np.mean(np.abs(vx[valid])))
                    max_abs_qy = float(np.max(np.abs(vy[valid])))
                    mean_abs_qy = float(np.mean(np.abs(vy[valid])))
                else:
                    max_abs_qx = mean_abs_qx = max_abs_qy = mean_abs_qy = np.nan

                sdet_data.update({
                    "cent_centers": cent_centers,
                    "qx_vals": vx,
                    "qy_vals": vy,
                    "qx_counts": cx,
                    "qy_counts": cy,
                    "max_abs_qx": max_abs_qx,
                    "mean_abs_qx": mean_abs_qx,
                    "max_abs_qy": max_abs_qy,
                    "mean_abs_qy": mean_abs_qy,
                })
                flat_scalars[f"Q_{sdet}_x_max_abs"] = max_abs_qx
                flat_scalars[f"Q_{sdet}_y_max_abs"] = max_abs_qy

                if not np.isnan(max_abs_qx) and max_abs_qx > max_recent_dev:
                    comp_dict["flags"].append(f"FLAG_RECENT_NONZERO_{sdet}")
                if not np.isnan(max_abs_qy) and max_abs_qy > max_recent_dev:
                    comp_dict["flags"].append(f"FLAG_RECENT_NONZERO_{sdet}")

        # 2. Twisting Covariance (N, S, NS)
        pxy_name = f"h_sEPD_Q_{sdet}_xy_{harmonic}_corr_avg"
        pxy = f.get(pxy_name)
        if pxy is not None:
            sdet_data["has_twisting"] = True
            edges = pxy.axis().edges()
            cent_centers = 0.5 * (edges[:-1] + edges[1:])
            vxy = pxy.values(flow=False)
            cxy = pxy.counts(flow=False)
            valid_xy = cxy > 0

            if np.any(valid_xy):
                max_abs_qxy = float(np.max(np.abs(vxy[valid_xy])))
                mean_abs_qxy = float(np.mean(np.abs(vxy[valid_xy])))
            else:
                max_abs_qxy = mean_abs_qxy = np.nan

            sdet_data.update({
                "cent_centers_xy": cent_centers,
                "qxy_vals": vxy,
                "qxy_counts": cxy,
                "max_abs_qxy": max_abs_qxy,
                "mean_abs_qxy": mean_abs_qxy,
            })
            flat_scalars[f"Q_{sdet}_xy_max_abs"] = max_abs_qxy

            if not np.isnan(max_abs_qxy) and max_abs_qxy > max_twist_dev:
                comp_dict["flags"].append(f"FLAG_TWIST_NONZERO_{sdet}")

        # 3. Flattening Variance Ratio (xx / yy == 1.0)
        pxx_name = f"h_sEPD_Q_{sdet}_xx_{harmonic}_corr_avg"
        pyy_name = f"h_sEPD_Q_{sdet}_yy_{harmonic}_corr_avg"
        pxx = f.get(pxx_name)
        pyy = f.get(pyy_name)
        if pxx is not None and pyy is not None:
            sdet_data["has_flattening_ratio"] = True
            edges = pxx.axis().edges()
            cent_centers = 0.5 * (edges[:-1] + edges[1:])
            vxx = pxx.values(flow=False)
            vyy = pyy.values(flow=False)
            cxx = pxx.counts(flow=False)
            cyy = pyy.counts(flow=False)

            valid_rat = (cxx > 0) & (cyy > 0) & (vyy != 0)
            ratios = np.full_like(vxx, np.nan, dtype=float)
            if np.any(valid_rat):
                ratios[valid_rat] = vxx[valid_rat] / vyy[valid_rat]
                ratio_dev = np.abs(ratios[valid_rat] - 1.0)
                max_dev_ratio = float(np.max(ratio_dev))
                mean_ratio = float(np.mean(ratios[valid_rat]))
            else:
                max_dev_ratio = mean_ratio = np.nan

            sdet_data.update({
                "cent_centers_rat": cent_centers,
                "qxx_vals": vxx,
                "qyy_vals": vyy,
                "qxx_yy_ratios": ratios,
                "max_dev_ratio_xxyy": max_dev_ratio,
                "mean_ratio_xxyy": mean_ratio,
            })
            flat_scalars[f"Q_{sdet}_xxyy_max_dev"] = max_dev_ratio

            if not np.isnan(max_dev_ratio) and max_dev_ratio > max_ratio_dev:
                comp_dict["flags"].append(f"FLAG_FLAT_ECCENTRICITY_{sdet}")

        comp_dict["subdetectors"][sdet] = sdet_data

    comp_dict["scalars"] = flat_scalars
    return comp_dict


def extract_qvec_run_metrics(
    file_path: Union[str, Path],
    subdetectors: Tuple[str, ...] = ("NS", "N", "S"),
    harmonic: int = 2,
    pass_suffix: str = "_corr2",
    cent_min: Optional[float] = None,
    cent_max: Optional[float] = None,
    cent_slices: Sequence[Tuple[float, float]] = (),
    syst_floor: float = 0.01,
    check_components: bool = True,
    cuts: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fast extraction of scalar flatness, harmonic, and intermediate Q-vector
    QA metrics from a single ROOT file for all specified subdetectors.

    The Psi distribution is projected over [cent_min, cent_max] (default: all centralities) and,
    separately, over each centrality slice in cent_slices (half-open [low, high) on bin centres),
    stored in record["slices"][label][subdet]. Statuses are set by evaluate_run() with the given
    cuts (DEFAULT_CUTS if None).

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
        slices: Dict[str, Dict[str, Any]] = {slice_label(s): {} for s in cent_slices}
        comp_metrics = None

        with uproot.open(path) as f_root:
            for subdet in subdetectors:
                hist_name = f"h2_sEPD_Psi_{subdet}_{harmonic}{pass_suffix}"
                values, edges, _, _ = get_qvec_psi_hist_data(
                    f_root,
                    hist_name=hist_name,
                    cent_min=cent_min,
                    cent_max=cent_max,
                )

                if values is None:
                    # Histogram missing
                    m = compute_psi_flatness_metrics(np.array([]), np.array([]))
                else:
                    m = compute_psi_flatness_metrics(values, edges, syst_floor=syst_floor)
                m["hist_name"] = hist_name
                subdet_metrics[subdet] = m

                if cent_slices:
                    arr = f_root[hist_name].to_numpy() if hist_name in f_root else None
                    for s in cent_slices:
                        if arr is not None and len(arr) == 3:
                            values_2d, edges_x, edges_y = arr
                            x_centers = 0.5 * (edges_x[:-1] + edges_x[1:])
                            mask = (x_centers >= s[0]) & (x_centers < s[1])
                            sm = compute_psi_flatness_metrics(values_2d[mask, :].sum(axis=0), edges_y, syst_floor=syst_floor)
                        else:
                            sm = compute_psi_flatness_metrics(np.array([]), np.array([]))
                        sm["hist_name"] = hist_name
                        slices[slice_label(s)][subdet] = sm

            # Intermediate Q-vector component checks (recentering, twisting, variance ratio);
            # their flags are set by evaluate_run() from the scalars
            if check_components:
                comp_metrics = extract_qvec_component_metrics(
                    f_root,
                    harmonic=harmonic,
                    subdetectors=subdetectors,
                )

        record: Dict[str, Any] = {
            "run_number": run_number,
            "file_path": str(path),
            "harmonic": harmonic,
            "pass_suffix": pass_suffix,
            "total_events": subdet_metrics.get("NS", {}).get("total_events", 0.0),
            "subdetectors": subdet_metrics,
        }
        if cent_slices:
            record["slices"] = slices

        if comp_metrics:
            record["components"] = comp_metrics
            record.update(comp_metrics.get("scalars", {}))

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

        evaluate_run(record, cuts if cuts is not None else DEFAULT_CUTS)
        return record, None

    except Exception as e:
        return None, f"Error processing {path.name}: {e}"
