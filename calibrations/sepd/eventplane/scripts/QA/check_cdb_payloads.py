#!/usr/bin/env python3
"""
check_cdb_payloads.py

Sanity checks on SEPD_EventPlaneCalib CDB payloads (one CDBTTree "Multiple" per run, one entry per
centrality bin IID), with an optional per-run comparison against a reference production.

Checks per payload:
- the file reads and every payload has the same IID (centrality bin) layout
- no NaN / inf values
- all-zero payloads (no bin filled: the calibration saw no events)
- all-zero bins other than the skipped bins (bin 0 is empty by design)
- second-moment matrices <Qxx>, <Qyy>, <Qxy> positive-definite in every filled bin
  (xx > 0, yy > 0, xx*yy - xy^2 > 0), as required for the flattening matrix

With --ref, compares each run present in both productions, bin by bin over bins filled in both:
max relative difference of <Qxx>/<Qyy>, max absolute difference of <Qx>/<Qy> (recentering) and <Qxy>.

Exits with status 1 if any payload fails a check.

Example:
  python3 check_cdb_payloads.py <production>/SEPD_EventPlaneCalib.list \\
      --ref <reference>/SEPD_EventPlaneCalib.list --out-dir <production>/cdb_check
"""

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import uproot

DETECTORS = ["N", "S", "NS"]
HARMONICS = [2, 3, 4]


def run_of(path: str) -> int:
    """Run number from a payload name, e.g. ...-73817.root or ...-73817v2.root."""
    m = re.search(r"-(\d+)(?:v\d+)?\.root$", path)
    if not m:
        raise ValueError(f"cannot parse run number from {path}")
    return int(m.group(1))


def load(path: str):
    """Returns the payload arrays, or the exception if it cannot be read."""
    try:
        return uproot.open(path)["Multiple"].arrays(library="np")
    except Exception as e:  # noqa: BLE001
        return e


def value_keys(a) -> list[str]:
    return [k for k in a if k != "IID"]


def check_payload(path: str, skip_bins: set[int]) -> dict:
    run = run_of(path)
    a = load(path)
    if isinstance(a, Exception):
        return {"run": run, "path": path, "error": f"read error: {a}"}

    iid = np.asarray(a["IID"])
    vals = np.vstack([a[k] for k in value_keys(a)])  # (n_values, n_bins)
    zero_bin = np.abs(vals).sum(axis=0) == 0
    checked = ~np.isin(iid, list(skip_bins))

    res = {
        "run": run,
        "path": path,
        "error": "",
        "arrays": a,
        "iids": tuple(iid.tolist()),
        "n_nan": int(np.isnan(vals).sum()),
        "n_inf": int(np.isinf(vals).sum()),
        "all_zero": bool(zero_bin.all()),
        "zero_bins": iid[zero_bin & checked].tolist(),
        "not_pd": [],
    }
    if res["all_zero"]:
        return res

    filled = ~zero_bin & checked
    for det in DETECTORS:
        for n in HARMONICS:
            keys = [f"DQ_{det}_{c}_{n}_avg" for c in ("xx", "yy", "xy")]
            if not all(k in a for k in keys):
                continue
            xx, yy, xy = (a[k] for k in keys)
            bad = filled & ~((xx > 0) & (yy > 0) & (xx * yy - xy * xy > 0))
            if bad.any():
                res["not_pd"].append(f"{det}{n}:bins {','.join(map(str, iid[bad]))}")
    return res


def compare(res: dict, ref_path: str) -> dict:
    b = load(ref_path)
    if isinstance(b, Exception):
        return {"run": res["run"], "note": f"reference read error: {b}"}
    a = res["arrays"]
    if not np.array_equal(a["IID"], b["IID"]):
        return {"run": res["run"], "note": "IID layout differs from reference"}

    va = np.vstack([a[k] for k in value_keys(a)])
    vb = np.vstack([b[k] for k in value_keys(a) if k in b]) if all(k in b for k in value_keys(a)) else None
    if vb is None:
        return {"run": res["run"], "note": "value keys differ from reference"}
    both = (np.abs(va).sum(axis=0) > 0) & (np.abs(vb).sum(axis=0) > 0)
    if not both.any():
        return {"run": res["run"], "note": "no bins filled in both"}

    out = {"run": res["run"], "note": "", "bins_compared": int(both.sum()),
           "second_moments_max_rel": 0.0, "recentering_max_abs": 0.0, "xy_max_abs": 0.0}
    for k in value_keys(a):
        x, y = a[k][both], b[k][both]
        if re.search(r"_(xx|yy)_\d_avg$", k):
            d = float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
            out["second_moments_max_rel"] = max(out["second_moments_max_rel"], d)
        elif re.search(r"_xy_\d_avg$", k):
            out["xy_max_abs"] = max(out["xy_max_abs"], float(np.max(np.abs(x - y))))
        elif re.search(r"_[xy]_\d_avg$", k):
            out["recentering_max_abs"] = max(out["recentering_max_abs"], float(np.max(np.abs(x - y))))
    return out


def summarize(col: np.ndarray, rows: list[dict]) -> str:
    i = int(np.argmax(col))
    return (f"median {np.median(col):.2e}  95% {np.percentile(col, 95):.2e}  "
            f"max {col[i]:.2e} (run {rows[i]['run']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity checks on SEPD_EventPlaneCalib CDB payloads.")
    parser.add_argument("payload_list", help="List of payload files (e.g. <production>/SEPD_EventPlaneCalib.list).")
    parser.add_argument("--ref", default=None, help="Payload list of a reference production to compare against.")
    parser.add_argument("--out-dir", default=None,
                        help="Write bad_payload_runs.list, all_zero_runs.list and (with --ref) payload_vs_ref.csv here.")
    parser.add_argument("--skip-bins", type=int, nargs="*", default=[0],
                        help="IID bins that are empty by design and not checked for zeros / positive-definiteness. Default: 0.")
    parser.add_argument("-j", "--workers", type=int, default=16, help="Parallel readers. Default: 16.")
    args = parser.parse_args()

    paths = [l.strip() for l in Path(args.payload_list).read_text().splitlines() if l.strip()]
    if not paths:
        print(f"Error: no payloads in {args.payload_list}", file=sys.stderr)
        return 1
    skip = set(args.skip_bins)

    with ThreadPoolExecutor(args.workers) as ex:
        results = sorted(ex.map(lambda p: check_payload(p, skip), paths), key=lambda r: r["run"])

    errors = [r for r in results if r["error"]]
    ok = [r for r in results if not r["error"]]
    layouts: dict[tuple, list[int]] = {}
    for r in ok:
        layouts.setdefault(r["iids"], []).append(r["run"])
    main_layout = max(layouts, key=lambda k: len(layouts[k])) if layouts else ()

    all_zero = [r["run"] for r in ok if r["all_zero"]]
    nonfinite = [r["run"] for r in ok if r["n_nan"] or r["n_inf"]]
    zero_bins = [(r["run"], r["zero_bins"]) for r in ok if not r["all_zero"] and r["zero_bins"]]
    not_pd = [(r["run"], r["not_pd"]) for r in ok if r["not_pd"]]
    odd_layout = [r["run"] for r in ok if r["iids"] != main_layout]

    print(f"Payloads: {len(results)}  (read errors: {len(errors)})")
    for r in errors[:10]:
        print(f"  {r['run']}: {r['error']}")
    if main_layout:
        print(f"IID layout: {len(main_layout)} bins ({main_layout[0]}..{main_layout[-1]}) in {len(layouts[main_layout])} payloads"
              + (f"; different layout in {len(odd_layout)}: {odd_layout[:10]}" if odd_layout else ""))
    print(f"Skipped bins (empty by design): {sorted(skip)}")
    print(f"All-zero payloads: {len(all_zero)}" + (f"  runs {all_zero[0]}..{all_zero[-1]}" if all_zero else ""))
    print(f"Payloads with NaN/inf: {len(nonfinite)}" + (f"  e.g. {nonfinite[:10]}" if nonfinite else ""))
    print(f"Payloads with zero bins (beyond skipped): {len(zero_bins)}")
    for run, bins in zero_bins[:10]:
        print(f"  {run}: bins {bins[:12]}")
    print(f"Payloads with non-positive-definite second moments: {len(not_pd)}")
    for run, d in not_pd[:10]:
        print(f"  {run}: {'; '.join(d)}")

    bad = sorted({r["run"] for r in errors} | set(all_zero) | set(nonfinite) | {r for r, _ in zero_bins}
                 | {r for r, _ in not_pd} | set(odd_layout))
    print(f"\nPayloads failing any check: {len(bad)} / {len(results)}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bad_payload_runs.list").write_text("".join(f"{r}\n" for r in bad))
        (out_dir / "all_zero_runs.list").write_text("".join(f"{r}\n" for r in all_zero))

    if args.ref:
        ref = {run_of(l.strip()): l.strip() for l in Path(args.ref).read_text().splitlines() if l.strip()}
        common = [r for r in ok if r["run"] in ref and not r["all_zero"]]
        print(f"\nReference: {args.ref}")
        print(f"Runs in both: {len([r for r in ok if r['run'] in ref])}  "
              f"(only here: {len([r for r in ok if r['run'] not in ref])}, only in reference: {len(set(ref) - {r['run'] for r in ok})})"
              f"; compared (not all-zero here): {len(common)}")
        with ThreadPoolExecutor(args.workers) as ex:
            comps = sorted(ex.map(lambda r: compare(r, ref[r["run"]]), common), key=lambda c: c["run"])
        rows = [c for c in comps if not c["note"]]
        notes = [c for c in comps if c["note"]]
        for c in notes[:10]:
            print(f"  {c['run']}: {c['note']}")
        if rows:
            identical = sum(c["second_moments_max_rel"] == 0 and c["recentering_max_abs"] == 0 and c["xy_max_abs"] == 0 for c in rows)
            print(f"Bit-identical to reference: {identical} / {len(rows)}")
            print("  <Qxx>,<Qyy> max rel diff:      " + summarize(np.array([c["second_moments_max_rel"] for c in rows]), rows))
            print("  <Qx>,<Qy> max abs diff:        " + summarize(np.array([c["recentering_max_abs"] for c in rows]), rows))
            print("  <Qxy> max abs diff:            " + summarize(np.array([c["xy_max_abs"] for c in rows]), rows))
        if out_dir:
            with open(out_dir / "payload_vs_ref.csv", "w") as f:
                f.write("run,bins_compared,second_moments_max_rel,recentering_max_abs,xy_max_abs,note\n")
                for c in comps:
                    if c["note"]:
                        f.write(f"{c['run']},,,,,{c['note']}\n")
                    else:
                        f.write(f"{c['run']},{c['bins_compared']},{c['second_moments_max_rel']:.3e},"
                                f"{c['recentering_max_abs']:.3e},{c['xy_max_abs']:.3e},\n")

    if out_dir:
        print(f"\nOutputs written to {out_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
