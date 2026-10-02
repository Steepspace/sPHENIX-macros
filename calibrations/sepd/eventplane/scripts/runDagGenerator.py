#!/usr/bin/env python3
"""
sEPD Q-Vector Calibration - HTCondor DAGMan Workflow Generator
==============================================================
This script automates the generation and submission of an HTCondor DAGMan
workflow for the sEPD Q-Vector Calibration pipeline.

Key Architecture:
-----------------
1. Per-Run Pipeline Independence:
   Each run executes through its own independent sub-DAG:
     [QA Segments] -> [Hadd Merge] -> [Pass 0] -> [Pass 1] -> [Pass 2] -> [Finalize]
   Eliminating global barriers allows fast runs to complete immediately without waiting
   for slower runs.

2. Simple Input Handling (Direct Input DST List):
   Like Jet-Vn/condor_utils, simply pass -i / --input-list pointing directly to a DST list
   or a text file containing a list of DST list paths. No CreateDstList.pl or complex
   directory synchronization required.

3. Multi-Submit-Node Balancing (Avoids the 15k Schedd Ceiling):
   HTCondor submit daemons (schedds) enforce a MAX_JOBS_RUNNING = 15,000 limit.
   When processing large datasets across many runs, this script can query 'condor_status -submitters',
   rank available submit nodes (sphnxuser01..08) by available headroom, and partition
   the runs across multiple submit nodes (using the same pattern as Jet-Vn/condor_utils).

4. Automatic Recovery & Concurrency Control:
   - Transient GPFS/network errors are automatically retried via RETRY directives.
   - Schedulers can run unthrottled (default) or capped with user-defined limits.
   - SCRIPT POST finalizes outputs per run into CDB/ and QVecCalib/ as soon as Pass 2 completes.

Author: Apurva Narde & Pair Programming Assistant
Date: 2026
"""

import argparse
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger("DAGGen")

SUBMISSION_NODES = [f"sphnxuser{i:02d}" for i in range(1, 9)]


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Central configuration container for the sEPD Calibration DAG workflow."""
    input_list: Path
    f4a_macro: Path
    f4a_QVecCalib: Path
    f4a_script: Path
    QVecCalib_script: Path
    output_dir: Path
    job_output_dir: Path | None
    condor_log_dir: Path

    dst_tag: str
    cdb_tag: str
    segments: int
    events: int
    f4a_condor_memory: float
    QVecCalib_condor_memory: float
    charge_threshold: float
    noise_threshold: float
    max_qa_jobs: int
    split_nodes: int
    submit_nodes: list[str]
    ranking_user: str
    verbose: bool

    @property
    def stage_qa_dir(self) -> Path:
        return self.output_dir / "stage-QA"

    @property
    def stage_calib_dir(self) -> Path:
        return self.output_dir / f"stage-QVecCalib-{self.charge_threshold}-{self.noise_threshold}"

    @property
    def dag_dir(self) -> Path:
        return self.output_dir / "dag"

    @property
    def submit_dir(self) -> Path:
        return self.output_dir / "submit"

    @property
    def scripts_dir(self) -> Path:
        return self.output_dir / "scripts"

    @property
    def qa_output_dir(self) -> Path:
        return self.output_dir / "QA"

    @property
    def final_qvec_dir(self) -> Path:
        return self.output_dir / "QVecCalib"

    @property
    def final_cdb_dir(self) -> Path:
        return self.output_dir / "CDB"

    def __str__(self):
        lines = ["DAG Workflow Configuration:"]
        max_len = max(len(f.name) for f in fields(self))
        for f in fields(self):
            lines.append(f"  {f.name:<{max_len}} : {getattr(self, f.name)}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------

def setup_logging(log_file: Path, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def run_command(command: list[str], cwd: Path | None = None) -> bool:
    logger.info(f"Executing: {' '.join(command)}")
    try:
        res = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        if res.stdout and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"STDOUT:\n{res.stdout.strip()}")
        if res.stderr:
            logger.error(f"STDERR:\n{res.stderr.strip()}")
        if res.returncode != 0:
            logger.error(f"Command failed with exit code {res.returncode}")
            return False
        return True
    except Exception as e:
        logger.critical(f"Error running command: {e}")
        return False


# -----------------------------------------------------------------------------
# Submit Node Load Balancing Logic (sphnxuser01..08)
# -----------------------------------------------------------------------------

def parse_submitters_output(output_text: str, user: str = "anarde") -> dict[str, dict[str, int]]:
    """
    Parses output of 'condor_status -submitters'.
    Aggregates user-specific and total RunningJobs and IdleJobs for sphnxuser01-08 nodes.
    """
    nodes = {
        node: {
            "user_running": 0,
            "user_idle": 0,
            "user_total": 0,
            "total_running": 0,
            "total_idle": 0,
        }
        for node in SUBMISSION_NODES
    }

    user_str = (user or "anarde").split("@")[0].lower()
    user_lower = user_str.split(".")[-1]

    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            m = re.search(r"(sphnxuser0[1-8])", parts[1], re.IGNORECASE)
            if m:
                node = m.group(1).lower()
                raw_user = parts[0].split("@")[0].lower()
                submitter_user = raw_user.split(".")[-1]
                try:
                    r_jobs = int(parts[2])
                    i_jobs = int(parts[3])
                except ValueError:
                    continue

                nodes[node]["total_running"] += r_jobs
                nodes[node]["total_idle"] += i_jobs

                if submitter_user == user_lower or raw_user == user_str:
                    nodes[node]["user_running"] += r_jobs
                    nodes[node]["user_idle"] += i_jobs
                    nodes[node]["user_total"] += (r_jobs + i_jobs)
    return nodes


def get_best_submit_nodes(user: str = "anarde") -> tuple[list[str], dict[str, dict[str, int]]]:
    """
    Ranks submission nodes (sphnxuser01-08) by running 'condor_status -submitters'.
    Sorted by lowest committed load (total_running + user_idle) to maximize headroom to the 15k limit.
    """
    nodes = {
        node: {
            "user_running": 0,
            "user_idle": 0,
            "user_total": 0,
            "total_running": 0,
            "total_idle": 0,
        }
        for node in SUBMISSION_NODES
    }
    try:
        res = subprocess.run(
            ["condor_status", "-submitters"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            nodes = parse_submitters_output(res.stdout, user=user)
    except Exception as e:
        logger.warning(f"Failed to query condor_status -submitters: {e}")

    ranked_nodes = sorted(
        SUBMISSION_NODES,
        key=lambda n: (
            nodes[n]["total_running"] + nodes[n]["user_idle"],
            nodes[n]["user_total"],
            nodes[n]["total_idle"],
            SUBMISSION_NODES.index(n),
        ),
    )
    return ranked_nodes, nodes


# -----------------------------------------------------------------------------
# Input Preparation & DST Splitting
# -----------------------------------------------------------------------------

def extract_run_number(path_or_str: str | Path) -> str:
    """Extracts the run number from a file name or path."""
    s = Path(path_or_str).name
    # Match patterns like -00068144, _00068144, -68144
    m = re.search(r"[-_]0*(\d{5,8})(?:[-_.]|$)", s)
    if m:
        return m.group(1).lstrip("0") or "0"
    m = re.search(r"\b0*(\d{5,8})\b", s)
    if m:
        return m.group(1).lstrip("0") or "0"
    return "0"


def prepare_dst_lists(config: PipelineConfig) -> dict[str, list[Path]]:
    """
    Parses the input list and splits DST files into per-segment .list files.

    Accepts:
    1. A list-of-lists file where each line is a path to a run's DST list
       (e.g., /path/to/dst_calofitting_zdc_sepd-00068144.list)
    2. A single DST list file directly where lines are the input data files
       (e.g., dst_calofitting_zdc_sepd-00068144.list)

    Returns:
        dict[str, list[Path]]: mapping run_number -> list of segment Path objects.
    """
    files_dir = config.stage_qa_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    input_lines = [line.strip() for line in config.input_list.read_text().splitlines() if line.strip()]
    if not input_lines:
        logger.critical(f"Input list {config.input_list} is empty!")
        sys.exit(1)

    # Check whether the input file is itself a single DST list or a list of DST list paths
    is_direct_dst_list = any(line.endswith(".root") or "," in line for line in input_lines[:5])

    dst_list_paths: list[Path] = []
    if is_direct_dst_list:
        logger.info(f"Input '{config.input_list.name}' detected as a single DST list.")
        dst_list_paths = [config.input_list]
    else:
        logger.info(f"Input '{config.input_list.name}' detected as a list of DST lists ({len(input_lines)} entries).")
        for line in input_lines:
            p = Path(line)
            if not p.is_absolute() and not p.is_file():
                # Check relative to input_list parent directory
                p_rel = (config.input_list.parent / p).resolve()
                p = p_rel if p_rel.is_file() else p.resolve()
            else:
                p = p.resolve()

            if not p.is_file():
                logger.warning(f"DST list file not found: {p}. Skipping.")
                continue
            dst_list_paths.append(p)

    run_segments: dict[str, list[Path]] = {}

    for dst_file in dst_list_paths:
        run = extract_run_number(dst_file.name)
        if not run or run == "0":
            # Fallback: check first line of file
            first_lines = dst_file.read_text().splitlines()
            if first_lines:
                run = extract_run_number(first_lines[0])

        if not run or run == "0":
            logger.warning(f"Could not parse run number from {dst_file}. Skipping.")
            continue

        stem = dst_file.stem
        try:
            lines = [l.strip() for l in dst_file.read_text().splitlines() if l.strip()]
            if config.segments > 0:
                lines = lines[:config.segments]
        except Exception as e:
            logger.error(f"Failed to read {dst_file}: {e}")
            continue

        run_segments[run] = []
        for i, line_str in enumerate(lines):
            seg_file = files_dir / f"{stem}-{i:03d}.list"
            seg_file.write_text(line_str + "\n")
            run_segments[run].append(seg_file)

        logger.info(f"Run {run} ({dst_file.name}): split into {len(run_segments[run])} segment(s).")

    return run_segments


# -----------------------------------------------------------------------------
# Template & Helper Script Generation
# -----------------------------------------------------------------------------

def write_helper_scripts(config: PipelineConfig) -> tuple[Path, Path]:
    """Writes the runHadd.sh and finalizeRun.sh helper scripts."""
    # 1. runHadd.sh
    hadd_sh = config.scripts_dir / "runHadd.sh"
    hadd_sh.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -e
        source /opt/sphenix/core/bin/sphenix_setup.sh -n new

        out_file="$1"
        in_dir="$2"

        if [ -z "$out_file" ] || [ -z "$in_dir" ]; then
            echo "Usage: $0 <output_qa_file> <input_hist_dir>" >&2
            exit 1
        fi

        mkdir -p "$(dirname "$out_file")"
        shopt -s nullglob
        files=("$in_dir"/*.root)

        if [ ${#files[@]} -eq 0 ]; then
            echo "Error: No .root files found in $in_dir" >&2
            exit 1
        fi

        echo "Merging ${#files[@]} histogram file(s) into $out_file..."
        hadd -f -n 11 "$out_file" "${files[@]}"
        echo "Merge completed successfully."
    """))
    hadd_sh.chmod(0o755)

    # 2. finalizeRun.sh (Executed as SCRIPT POST after Pass 2)
    finalize_sh = config.scripts_dir / "finalizeRun.sh"
    finalize_sh.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        # Post-script to finalize calibrated outputs for a single run
        ret_code="$1"
        run="$2"
        pass2_out="$3"
        final_qvec="$4"
        final_cdb="$5"

        if [ "$ret_code" -ne 0 ]; then
            echo "PASS_2 failed with exit code $ret_code. Skipping finalization for Run $run." >&2
            exit "$ret_code"
        fi

        mkdir -p "$final_qvec" "$final_cdb"

        # Copy final calibrated QA root file
        if [ -f "$pass2_out/hist/QVecCalib-$run.root" ]; then
            cp -v "$pass2_out/hist/QVecCalib-$run.root" "$final_qvec/"
        else
            echo "Warning: $pass2_out/hist/QVecCalib-$run.root not found." >&2
        fi

        # Copy final CDB payloads
        if [ -d "$pass2_out/CDB/$run" ]; then
            cp -rv "$pass2_out/CDB/$run" "$final_cdb/"
        else
            echo "Warning: $pass2_out/CDB/$run not found." >&2
        fi

        echo ">>> Run $run successfully calibrated and finalized!"
        exit 0
    """))
    finalize_sh.chmod(0o755)

    return hadd_sh, finalize_sh


def write_condor_submit_templates(
    config: PipelineConfig,
    hadd_sh: Path,
) -> None:
    """Generates the generic Condor submit files for QA, Hadd, and Calibration passes."""
    # 1. qa.sub
    (config.submit_dir / "qa.sub").write_text(textwrap.dedent(f"""\
        executable     = {config.f4a_script}
        arguments      = {config.f4a_macro} $(input_dst) test-$(run)-$(seg).root tree-$(run)-$(seg).root {config.events} {config.cdb_tag} {config.stage_qa_dir}/output
        log            = {config.condor_log_dir}/qa-$(run)-$(seg).log
        output         = {config.stage_qa_dir}/stdout/qa-$(run)-$(seg).out
        error          = {config.stage_qa_dir}/error/qa-$(run)-$(seg).err
        request_memory = {config.f4a_condor_memory}GB
        queue
    """))

    # 2. hadd.sub
    (config.submit_dir / "hadd.sub").write_text(textwrap.dedent(f"""\
        executable     = {hadd_sh}
        arguments      = $(output_qa) $(input_hist_dir)
        log            = {config.condor_log_dir}/hadd-$(run).log
        output         = {config.stage_qa_dir}/stdout/hadd-$(run).out
        error          = {config.stage_qa_dir}/error/hadd-$(run).err
        request_memory = 2GB
        queue
    """))

    # 3. calib.sub (Shared for Pass 0, 1, 2)
    (config.submit_dir / "calib.sub").write_text(textwrap.dedent(f"""\
        executable     = {config.QVecCalib_script}
        arguments      = {config.f4a_QVecCalib} $(tree_dir) $(qa_hist) $(calib_hist) $(pass_num) {config.charge_threshold} {config.noise_threshold} {config.dst_tag} $(pass_out_dir)
        log            = {config.condor_log_dir}/calib-$(run)-p$(pass_num).log
        output         = $(calib_type_dir)/stdout/calib-$(run)-p$(pass_num).out
        error          = $(calib_type_dir)/error/calib-$(run)-p$(pass_num).err
        request_memory = {config.QVecCalib_condor_memory}GB
        queue
    """))


# -----------------------------------------------------------------------------
# DAG Workflow Generation
# -----------------------------------------------------------------------------

def generate_dags(
    config: PipelineConfig,
    run_segments: dict[str, list[Path]],
    finalize_sh: Path,
    target_nodes: list[str],
) -> dict[str, Path]:
    """
    Generates modular per-run DAGs and master DAG(s).
    If target_nodes has multiple nodes, partitions runs across them.
    Returns a dict mapping node_name -> master_dag_path.
    """
    # DAGMan config: ensure fast and responsive scheduling with high idle headroom
    dagman_config = config.dag_dir / "dagman.config"
    dagman_config.write_text(textwrap.dedent("""\
        DAGMAN_MAX_JOBS_SUBMITTED = 0
        DAGMAN_MAX_JOBS_IDLE = 5000
        DAGMAN_SUBMIT_DELAY = 0
    """))

    pass0_dir = config.stage_calib_dir / "ComputeRecentering"
    pass1_dir = config.stage_calib_dir / "ApplyRecentering"
    pass2_dir = config.stage_calib_dir / "ApplyFlattening"

    pass0_out = pass0_dir / "output"
    pass1_out = pass1_dir / "output"
    pass2_out = pass2_dir / "output"

    active_runs = [r for r, segs in run_segments.items() if segs]

    # Generate individual run DAGs
    for run in active_runs:
        seg_files = run_segments[run]
        run_dag_path = config.dag_dir / f"run_{run}.dag"
        dag_lines = [
            f"# =====================================================================",
            f"# Workflow for Run {run} ({len(seg_files)} segments)",
            f"# =====================================================================",
            "",
            "# --- Stage 1: QA Segment Production ---",
        ]

        qa_node_names = []
        for i, seg_file in enumerate(seg_files):
            seg_str = f"{i:03d}"
            node_name = f"QA_{seg_str}"
            qa_node_names.append(node_name)

            dag_lines.append(f'JOB {node_name} ../submit/qa.sub')
            dag_lines.append(f'VARS {node_name} run="{run}" seg="{seg_str}" input_dst="{seg_file.resolve()}"')
            if config.max_qa_jobs > 0:
                dag_lines.append(f'CATEGORY {node_name} QA_LIMIT')
            dag_lines.append(f'RETRY {node_name} 3')

        # --- Stage 2: Hadd Merge ---
        output_qa = config.qa_output_dir / f"QA-{run}.root"
        input_hist_dir = config.stage_qa_dir / "output" / run / "hist"

        dag_lines.extend([
            "",
            "# --- Stage 2: Parallel QA Histogram Merge ---",
            f"JOB HADD ../submit/hadd.sub",
            f'VARS HADD run="{run}" output_qa="{output_qa}" input_hist_dir="{input_hist_dir}"',
            f'PARENT {" ".join(qa_node_names)} CHILD HADD',
            f"RETRY HADD 2",
        ])

        # --- Stage 3: Iterative Calibration Passes ---
        tree_dir = config.stage_qa_dir / "output" / run / "tree"
        pass0_calib = pass0_out / "hist" / f"QVecCalib-{run}.root"
        pass1_calib = pass1_out / "hist" / f"QVecCalib-{run}.root"

        dag_lines.extend([
            "",
            "# --- Stage 3: Iterative Calibration Passes ---",
            "# Pass 0: ComputeRecentering",
            f"JOB PASS_0 ../submit/calib.sub",
            f'VARS PASS_0 run="{run}" pass_num="0" tree_dir="{tree_dir}" qa_hist="{output_qa}" calib_hist="none" pass_out_dir="{pass0_out}" calib_type_dir="{pass0_dir}"',
            f"PARENT HADD CHILD PASS_0",
            f"RETRY PASS_0 2",
            "",
            "# Pass 1: ApplyRecentering",
            f"JOB PASS_1 ../submit/calib.sub",
            f'VARS PASS_1 run="{run}" pass_num="1" tree_dir="{tree_dir}" qa_hist="{output_qa}" calib_hist="{pass0_calib}" pass_out_dir="{pass1_out}" calib_type_dir="{pass1_dir}"',
            f"PARENT PASS_0 CHILD PASS_1",
            f"RETRY PASS_1 2",
            "",
            "# Pass 2: ApplyFlattening",
            f"JOB PASS_2 ../submit/calib.sub",
            f'VARS PASS_2 run="{run}" pass_num="2" tree_dir="{tree_dir}" qa_hist="{output_qa}" calib_hist="{pass1_calib}" pass_out_dir="{pass2_out}" calib_type_dir="{pass2_dir}"',
            f"PARENT PASS_1 CHILD PASS_2",
            f"RETRY PASS_2 2",
            "",
            "# Finalize Run Output as soon as Pass 2 completes",
            f"SCRIPT POST PASS_2 {finalize_sh} $RETURN {run} {pass2_out} {config.final_qvec_dir} {config.final_cdb_dir}",
            "",
        ])

        run_dag_path.write_text("\n".join(dag_lines) + "\n")

    # Generate Partitioned Master DAGs
    node_dags: dict[str, Path] = {}
    node_run_map: dict[str, list[str]] = {node: [] for node in target_nodes}

    for idx, run in enumerate(active_runs):
        target_node = target_nodes[idx % len(target_nodes)]
        node_run_map[target_node].append(run)

    for node, runs in node_run_map.items():
        if not runs:
            continue
        dag_filename = f"master_{node}.dag" if len(target_nodes) > 1 else "master.dag"
        node_dag_path = config.dag_dir / dag_filename

        lines = [
            "# =====================================================================",
            f"# Master DAG for submit node: {node} ({len(runs)} runs)",
            "# =====================================================================",
            f"CONFIG {config.dag_dir}/dagman.config",
            "",
        ]
        for run in runs:
            lines.append(f"SPLICE RUN_{run} run_{run}.dag")

        if config.max_qa_jobs > 0:
            lines.extend([
                "",
                "# Optional Concurrency Throttle",
                f"MAXJOBS QA_LIMIT {config.max_qa_jobs}",
            ])

        node_dag_path.write_text("\n".join(lines) + "\n")
        node_dags[node] = node_dag_path

    # Always generate a complete master.dag containing all runs as fallback
    if len(target_nodes) > 1:
        all_dag_path = config.dag_dir / "master.dag"
        all_lines = [
            "# =====================================================================",
            f"# Unified Master DAG (All {len(active_runs)} runs)",
            "# =====================================================================",
            f"CONFIG {config.dag_dir}/dagman.config",
            "",
        ]
        for run in active_runs:
            all_lines.append(f"SPLICE RUN_{run} run_{run}.dag")
        if config.max_qa_jobs > 0:
            all_lines.extend(["", f"MAXJOBS QA_LIMIT {config.max_qa_jobs}"])
        all_dag_path.write_text("\n".join(all_lines) + "\n")

    return node_dags


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="sEPD Calibration HTCondor DAG Workflow Generator (Multi-Submit-Node Scalable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req_grp = parser.add_argument_group("Required Arguments")
    req_grp.add_argument("-i", "--input-list", "--input", dest="input_list", type=str, required=True,
                         help="Input DST list (or text file containing list of DST list paths).")

    opt_grp = parser.add_argument_group("Macros & Scripts")
    opt_grp.add_argument("-i2", "--f4a-macro", type=str, default="macros/Fun4All_sEPD.C")
    opt_grp.add_argument("-i3", "--f4a-QVecCalib", type=str, default="macros/Fun4All_QVecCalib.C")
    opt_grp.add_argument("-e1", "--f4a-script", type=str, default="scripts/genFun4All.sh")
    opt_grp.add_argument("-e2", "--QVecCalib-script", type=str, default="scripts/genQVecCalib.sh")

    param_grp = parser.add_argument_group("Parameters & Thresholds")
    param_grp.add_argument("-n1", "--segments", type=int, default=50, help="Segments per run to process (0 = all).")
    param_grp.add_argument("-n2", "--events", type=int, default=0, help="Events per job (0 = all).")
    param_grp.add_argument("-t1", "--dst-tag", type=str, default="pro001_pcdb001_v001", help="Production tag.")
    param_grp.add_argument("-t2", "--cdb-tag", type=str, default="newcdbtag", help="CDB tag.")
    param_grp.add_argument("-c1", "--charge-threshold", type=float, default=50.0)
    param_grp.add_argument("-c2", "--noise-threshold", type=float, default=0.5)

    res_grp = parser.add_argument_group("Condor Resources & Load Balancing")
    res_grp.add_argument("-m1", "--f4a-memory", type=float, default=1.0, help="Memory for QA jobs in GB.")
    res_grp.add_argument("-m2", "--calib-memory", "--QVecCalib-memory", type=float, default=0.5, help="Memory for calib jobs in GB.")
    res_grp.add_argument("-l", "--condor-log-dir", type=str, default="")
    res_grp.add_argument("--max-qa-jobs", type=int, default=0, help="Maximum concurrent QA jobs in DAG (0 = unconstrained up to schedd limits).")
    res_grp.add_argument("--split-nodes", type=int, default=1, help="Distribute runs across N best submit nodes (1 to 8) to avoid the 15k job schedd cap.")
    res_grp.add_argument("--submit-nodes", type=str, default="", help="Explicit comma-separated list of submit nodes (e.g. sphnxuser01,sphnxuser02).")
    res_grp.add_argument("--user", type=str, default="", help="Username for submit node ranking (default: current $USER).")

    out_grp = parser.add_argument_group("Output & Submission")
    out_grp.add_argument("-o", "--output", type=str, default="test", help="Project output directory.")
    out_grp.add_argument("-o2", "--job-output-dir", "--job-output", type=str, default=None, help="Alternate output dir for raw tree/hist files.")
    out_grp.add_argument("--submit", action="store_true", help="Automatically submit generated DAG(s) across target nodes.")
    out_grp.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")

    args = parser.parse_args()

    # Paths resolution
    output_dir = Path(args.output).resolve()
    input_list = Path(args.input_list).resolve()

    current_user = os.environ.get("USER", "unknown")
    ranking_user = args.user if args.user else current_user
    condor_log_dir = Path(args.condor_log_dir).resolve() if args.condor_log_dir else Path(f"/tmp/{current_user}/condor_logs")

    # Determine submit nodes
    if args.submit_nodes:
        target_nodes = [n.strip().lower() for n in args.submit_nodes.split(",") if n.strip()]
    elif args.split_nodes > 1:
        ranked_nodes, node_stats = get_best_submit_nodes(user=ranking_user)
        n_split = min(args.split_nodes, len(ranked_nodes))
        target_nodes = ranked_nodes[:n_split]
    else:
        # Default single node: current host if recognized, else best ranked
        current_host = socket.gethostname().split(".")[0].lower()
        if current_host in SUBMISSION_NODES:
            target_nodes = [current_host]
        else:
            ranked_nodes, _ = get_best_submit_nodes(user=ranking_user)
            target_nodes = [ranked_nodes[0]]

    config = PipelineConfig(
        input_list=input_list,
        f4a_macro=Path(args.f4a_macro).resolve(),
        f4a_QVecCalib=Path(args.f4a_QVecCalib).resolve(),
        f4a_script=Path(args.f4a_script).resolve(),
        QVecCalib_script=Path(args.QVecCalib_script).resolve(),
        output_dir=output_dir,
        job_output_dir=Path(args.job_output_dir).resolve() if args.job_output_dir else None,
        condor_log_dir=condor_log_dir,
        dst_tag=args.dst_tag,
        cdb_tag=args.cdb_tag,
        segments=args.segments,
        events=args.events,
        f4a_condor_memory=args.f4a_memory,
        QVecCalib_condor_memory=args.calib_memory,
        charge_threshold=args.charge_threshold,
        noise_threshold=args.noise_threshold,
        max_qa_jobs=args.max_qa_jobs,
        split_nodes=args.split_nodes,
        submit_nodes=target_nodes,
        ranking_user=ranking_user,
        verbose=args.verbose,
    )

    # Validation
    for f in [
        config.input_list,
        config.f4a_macro,
        config.f4a_QVecCalib,
        config.f4a_script,
        config.QVecCalib_script,
    ]:
        if not f.is_file():
            print(f"Error: Missing required input file: {f}", file=sys.stderr)
            sys.exit(1)

    # Directory Setup
    directories = [
        config.output_dir,
        config.dag_dir,
        config.submit_dir,
        config.scripts_dir,
        config.condor_log_dir,
        config.qa_output_dir,
        config.final_qvec_dir,
        config.final_cdb_dir,
        config.stage_qa_dir / "stdout",
        config.stage_qa_dir / "error",
    ]

    for calib in ["ComputeRecentering", "ApplyRecentering", "ApplyFlattening"]:
        for subdir in ["stdout", "error", "output", "output/hist"]:
            directories.append(config.stage_calib_dir / calib / subdir)
    directories.append(config.stage_calib_dir / "ApplyFlattening" / "output" / "CDB")

    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    # Setup Logging
    log_file = config.output_dir / "dag_generator.log"
    setup_logging(log_file, config.verbose)

    logger.info("=" * 65)
    logger.info(config)
    logger.info(f"Target Submit Nodes: {target_nodes}")
    logger.info("=" * 65)

    # Handle alternate output directory symlink for stage-QA
    qa_out_symlink = config.stage_qa_dir / "output"
    if config.job_output_dir:
        config.job_output_dir.mkdir(parents=True, exist_ok=True)
        if qa_out_symlink.is_symlink() or qa_out_symlink.is_file():
            qa_out_symlink.unlink()
        elif qa_out_symlink.is_dir():
            shutil.rmtree(qa_out_symlink)
        qa_out_symlink.symlink_to(config.job_output_dir, target_is_directory=True)
    else:
        if qa_out_symlink.is_symlink() or qa_out_symlink.is_file():
            qa_out_symlink.unlink()
        qa_out_symlink.mkdir(parents=True, exist_ok=True)

    # 1. Prepare DSTs directly from input
    run_segments = prepare_dst_lists(config)
    total_segments = sum(len(segs) for segs in run_segments.values())
    logger.info(f"Prepared {len(run_segments)} runs with {total_segments} total segment jobs.")

    # 2. Helper scripts and submit templates
    hadd_sh, finalize_sh = write_helper_scripts(config)
    write_condor_submit_templates(config, hadd_sh)

    # 3. Generate DAGs (Partitioned across target nodes)
    node_dags = generate_dags(config, run_segments, finalize_sh, target_nodes)

    log_dir = config.condor_log_dir or (config.output_dir / "logs")
    prep_cmd = f"rm -rf {log_dir} && mkdir -p {log_dir} && "

    commands = []
    for node, dag_file in node_dags.items():
        base_cmd = f"{prep_cmd}cd {config.dag_dir} && condor_submit_dag {dag_file.name}"
        cmd = f"ssh {node} '{base_cmd}'"
        commands.append((node, dag_file, cmd))

    if not args.submit:
        print("\n" + "=" * 65)
        print("HTCondor DAGMan Workflow Successfully Generated!")
        print("=" * 65)
        print(f"Total Runs: {len(run_segments)} | Total Segments: {total_segments}")
        print(f"Distributed across {len(target_nodes)} submit node(s): {', '.join(target_nodes)}")
        print("=" * 65)
        print("\nTo submit jobs:")
        for node, dag_file, cmd in commands:
            print(f"  {cmd}")
        print("\nTo monitor progress:")
        for node, dag_file, _ in commands:
            print(f"  ssh {node} condor_q -dag")
            print(f"  tail -f {config.dag_dir}/{dag_file.name}.dagman.out")
        print("=" * 65 + "\n")
    else:
        logger.info("Submitting DAGs across target nodes...")
        for node, dag_file, cmd in commands:
            logger.info(f"Submitting on {node}: {cmd}")
            run_command(["bash", "-c", cmd])


if __name__ == "__main__":
    main()
