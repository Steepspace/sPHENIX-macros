#!/usr/bin/env python3
"""
sEPD Calibration Pipeline - HTCondor DAGMan Workflow Monitor
============================================================
Monitors an active sEPD calibration DAGMan workflow (or partitioned multi-node DAGs)
until all runs and stages reach completion, then optionally dispatches a summary email.

Key Capabilities:
-----------------
1. Multi-DAG & Multi-Submit-Node Aware:
   Automatically detects single master.dag or partitioned master_<node>.dag files
   and aggregates node progress across all submit nodes.

2. Real-Time DAGMan Introspection:
   Parses live DAGMan status directly from .dagman.out and lock files:
   - Node status: Total, Completed, Queued/Running, Ready, Unready, Failed, Held.
   - DAG status: RUNNING, COMPLETED, FAILED.

3. Run-Level Completion Tracking:
   Cross-references final outputs in CDB/ and QVecCalib/ to report exactly which
   runs have finalized and which are pending or errored.

4. Robust Email Notification:
   Sends a detailed completion report via local mailx/mail/sendmail or SMTP.

Usage Examples:
---------------
  # Run once and view status dashboard:
  python scripts/monitor_dag.py /path/to/output_dir --run-once

  # Monitor live every 60s and send email on completion:
  python scripts/monitor_dag.py /path/to/output_dir 60s user@bnl.gov

  # Using flags:
  python scripts/monitor_dag.py -d /path/to/output_dir -i 2m -e user@bnl.gov

  # Test email configuration:
  python scripts/monitor_dag.py --test-email -e user@bnl.gov
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import email.utils
from email.mime.text import MIMEText
import fnmatch
import math
import os
from pathlib import Path
import re
import shutil
import smtplib
import socket
import subprocess
import sys
import time

# Ensure unbuffered output for live tailing in logs or nohup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class DagNodeStats:
    total: int = 0
    done: int = 0
    pre: int = 0
    queued: int = 0
    post: int = 0
    ready: int = 0
    unready: int = 0
    failed: int = 0
    futile: int = 0
    held: int = 0


@dataclass
class DagStatus:
    name: str
    dag_path: Path
    out_path: Path
    lock_path: Path
    sub_path: Path
    is_running: bool = False
    exit_status: int | None = None
    stats: DagNodeStats = field(default_factory=DagNodeStats)
    last_event_time: str = ""


@dataclass
class PipelineStatus:
    output_dir: Path
    dag_dir: Path
    dags: list[DagStatus] = field(default_factory=list)
    total_runs: int = 0
    finalized_runs: list[str] = field(default_factory=list)
    pending_runs: list[str] = field(default_factory=list)
    active_errors: list[str] = field(default_factory=list)

    @property
    def is_all_complete(self) -> bool:
        if not self.dags:
            return False
        return all(not d.is_running and d.exit_status is not None for d in self.dags)

    @property
    def is_success(self) -> bool:
        if not self.is_all_complete:
            return False
        return all(d.exit_status == 0 and d.stats.failed == 0 for d in self.dags)

    @property
    def aggregate_stats(self) -> DagNodeStats:
        agg = DagNodeStats()
        for d in self.dags:
            agg.total += d.stats.total
            agg.done += d.stats.done
            agg.pre += d.stats.pre
            agg.queued += d.stats.queued
            agg.post += d.stats.post
            agg.ready += d.stats.ready
            agg.unready += d.stats.unready
            agg.failed += d.stats.failed
            agg.futile += d.stats.futile
            agg.held += d.stats.held
        return agg


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def parse_interval(val: str) -> float:
    """Parses intervals like '30s', '2m', '1h', '60' into seconds."""
    val = val.strip()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)$", val)
    if not match:
        raise ValueError(f"Invalid interval: '{val}'. Expected e.g. '30s', '2m', '1h', '60'.")
    num = float(match.group(1))
    unit = match.group(2).lower()
    if unit in ("", "s", "sec", "second", "seconds"):
        return num
    if unit in ("m", "min", "minute", "minutes"):
        return num * 60.0
    if unit in ("h", "hr", "hour", "hours"):
        return num * 3600.0
    if unit in ("d", "day", "days"):
        return num * 86400.0
    raise ValueError(f"Unknown time unit '{unit}' in interval: '{val}'")


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable string like '1h 23m 45s'."""
    sec_int = max(0, int(seconds))
    hours, remainder = divmod(sec_int, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def make_progress_bar(done: int, total: int, width: int = 30) -> str:
    """Creates a visual ASCII/Unicode progress bar."""
    if total <= 0:
        return "[" + "-" * width + "] 0.0%"
    pct = min(100.0, max(0.0, done / total * 100.0))
    filled = int(round(width * (pct / 100.0)))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


# -----------------------------------------------------------------------------
# DAG Output Parsing
# -----------------------------------------------------------------------------

def parse_dagman_out(out_path: Path, max_bytes: int = 131072) -> tuple[DagNodeStats, int | None, str]:
    """
    Parses the tail of a .dagman.out log file to extract the latest status table,
    held job count, and exit status.
    """
    stats = DagNodeStats()
    exit_status = None
    last_event_time = ""

    if not out_path.is_file():
        return stats, exit_status, last_event_time

    try:
        size = out_path.stat().st_size
        if size == 0:
            return stats, exit_status, last_event_time

        read_size = min(size, max_bytes)
        with open(out_path, "rb") as f:
            if size > read_size:
                f.seek(size - read_size)
            raw = f.read(read_size)

        text = raw.decode("utf-8", errors="ignore")
        raw_lines = text.splitlines()

        # Clean timestamp/daemon prefix: "10/03/26 23:49:11.767 (D_ALWAYS) "
        clean_lines = []
        for line in raw_lines:
            m = re.match(r"^(\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+\(D_[A-Z]+\)\s*(.*)$", line)
            if m:
                last_event_time = m.group(1)
                clean_lines.append(m.group(2))
            else:
                clean_lines.append(line)

        clean_text = "\n".join(clean_lines)

        # 1. Search for latest table block:
        # Of 540 nodes total:
        #  Done     Pre   Queued    Post   Ready   Un-Ready   Failed   Futile
        #   ===     ===      ===     ===     ===        ===      ===      ===
        #   536       0        3       0       0          1        0        0
        table_pattern = (
            r"Of\s+(\d+)\s+nodes total:\s*\n"
            r"\s*Done\s+Pre\s+Queued\s+Post\s+Ready\s+Un-?Ready\s+Failed\s+Futile\s*\n"
            r"\s*===.*?\n"
            r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
        matches = list(re.finditer(table_pattern, clean_text, re.IGNORECASE))
        if matches:
            last = matches[-1]
            stats.total = int(last.group(1))
            stats.done = int(last.group(2))
            stats.pre = int(last.group(3))
            stats.queued = int(last.group(4))
            stats.post = int(last.group(5))
            stats.ready = int(last.group(6))
            stats.unready = int(last.group(7))
            stats.failed = int(last.group(8))
            stats.futile = int(last.group(9))

        # 2. Check held jobs
        held_matches = list(re.finditer(r"(\d+)\s+job proc\(s\)\s+currently held", clean_text, re.IGNORECASE))
        if held_matches:
            stats.held = int(held_matches[-1].group(1))

        # 3. Check exit status
        exit_matches = list(re.finditer(r"EXITING WITH STATUS\s+(\d+)", clean_text, re.IGNORECASE))
        if exit_matches:
            exit_status = int(exit_matches[-1].group(1))

    except Exception:
        pass

    return stats, exit_status, last_event_time


def inspect_pipeline(target_path: Path) -> PipelineStatus:
    """
    Inspects the target output or dag directory and aggregates status
    across all DAGs and final output directories.
    """
    target = target_path.expanduser().resolve()
    if target.name == "dag":
        dag_dir = target
        output_dir = target.parent
    else:
        output_dir = target
        dag_dir = target / "dag"

    status = PipelineStatus(output_dir=output_dir, dag_dir=dag_dir)

    if not dag_dir.is_dir():
        return status

    # Find master DAG files:
    # 1. master_*.dag (partitioned)
    # 2. master.dag (unified / single)
    partitioned_dags = sorted(dag_dir.glob("master_*.dag"))
    if partitioned_dags:
        dag_files = partitioned_dags
    else:
        master_dag = dag_dir / "master.dag"
        dag_files = [master_dag] if master_dag.is_file() else []

    # Inspect each DAG
    for dag_p in dag_files:
        name = dag_p.name
        out_p = dag_p.with_name(f"{name}.dagman.out")
        lock_p = dag_p.with_name(f"{name}.lock")
        sub_p = dag_p.with_name(f"{name}.condor.sub")

        stats, exit_code, last_time = parse_dagman_out(out_p)
        is_running = lock_p.is_file()

        # If lock file doesn't exist, but exit status was found, DAG has finished
        if not is_running and exit_code is None and out_p.is_file():
            # Check if DAGMan logged complete termination
            pass

        d_status = DagStatus(
            name=name,
            dag_path=dag_p,
            out_path=out_p,
            lock_path=lock_p,
            sub_path=sub_p,
            is_running=is_running,
            exit_status=exit_code,
            stats=stats,
            last_event_time=last_time,
        )
        status.dags.append(d_status)

    # Inspect runs from dag/runs directory or dag_generator.log
    runs_dir = dag_dir / "runs"
    all_runs = set()
    if runs_dir.is_dir():
        for f in runs_dir.glob("run_*.dag"):
            m = re.match(r"^run_(\d+)\.dag$", f.name)
            if m:
                all_runs.add(m.group(1))

    status.total_runs = len(all_runs)

    # Inspect finalized runs in CDB/ and QVecCalib/
    cdb_dir = output_dir / "CDB"
    qvec_dir = output_dir / "QVecCalib"
    finalized = set()

    if cdb_dir.is_dir():
        for entry in cdb_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                finalized.add(entry.name)

    if qvec_dir.is_dir():
        for entry in qvec_dir.glob("QVecCalib-*.root"):
            m = re.match(r"^QVecCalib-(\d+)\.root$", entry.name)
            if m:
                finalized.add(m.group(1))

    if all_runs:
        status.finalized_runs = sorted(finalized.intersection(all_runs))
        status.pending_runs = sorted(all_runs - finalized)
    else:
        status.finalized_runs = sorted(finalized)
        status.total_runs = len(status.finalized_runs)

    # Scan for recent non-empty error files
    err_samples = []
    for err_dir_pattern in [
        output_dir / "stage-QA" / "error",
        output_dir / "stage-QVecCalib-*" / "*" / "error",
    ]:
        for err_d in [output_dir] if err_dir_pattern == output_dir else Path(output_dir).glob(str(err_dir_pattern.relative_to(output_dir))):
            if err_d.is_dir():
                for ef in err_d.glob("*.err"):
                    try:
                        if ef.stat().st_size > 0:
                            err_samples.append(ef.name)
                            if len(err_samples) >= 10:
                                break
                    except OSError:
                        pass
    status.active_errors = err_samples

    return status


# -----------------------------------------------------------------------------
# Terminal Dashboard Display
# -----------------------------------------------------------------------------

def print_dashboard(
    status: PipelineStatus,
    start_time: datetime,
    check_count: int,
    interval_sec: float,
    clear_screen: bool = False,
) -> None:
    """Renders a structured, readable terminal dashboard."""
    if clear_screen and sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")

    now = datetime.now()
    elapsed = format_duration((now - start_time).total_seconds())
    agg = status.aggregate_stats

    print("=" * 78)
    print(f" sEPD Calibration Pipeline Monitor | {socket.gethostname()} | Elapsed: {elapsed}")
    print("=" * 78)
    print(f"Project Output: {status.output_dir}")
    print(f"DAG Directory:  {status.dag_dir}")
    print(f"Checks Done:    {check_count} (Interval: {interval_sec:.0f}s) | Last Update: {now.strftime('%H:%M:%S')}")
    print("-" * 78)

    # 1. Overall Workflow Progress
    progress_bar = make_progress_bar(agg.done, agg.total, width=32)
    print(f"DAG Node Progress:  {progress_bar} ({agg.done}/{agg.total} nodes)")

    run_pct = (len(status.finalized_runs) / status.total_runs * 100.0) if status.total_runs > 0 else 0.0
    run_bar = make_progress_bar(len(status.finalized_runs), status.total_runs, width=32)
    print(f"Finalized Runs:     {run_bar} ({len(status.finalized_runs)}/{status.total_runs} runs)")

    # 2. Detailed Node Status Table
    print("\nNode Counts Summary:")
    print(f"  Done:     {agg.done:<7} | Queued:   {agg.queued:<7} | Ready:   {agg.ready:<7}")
    print(f"  Unready:  {agg.unready:<7} | Failed:   {agg.failed:<7} | Held:    {agg.held:<7}")

    # 3. Per-DAG Breakdown (especially for multi-node partitioned DAGs)
    if len(status.dags) > 1:
        print("\nPartitioned DAG Breakdown:")
        print(f"  {'DAG Name':<28} {'State':<12} {'Done':<8} {'Queued':<8} {'Failed':<8} {'Exit'}")
        print("  " + "-" * 70)
        for d in status.dags:
            if d.is_running:
                state_str = "RUNNING"
            elif d.exit_status == 0:
                state_str = "DONE (OK)"
            elif d.exit_status is not None:
                state_str = f"EXIT({d.exit_status})"
            else:
                state_str = "IDLE / WAIT"

            exit_str = str(d.exit_status) if d.exit_status is not None else "-"
            print(f"  {d.name:<28} {state_str:<12} {d.stats.done:<8} {d.stats.queued:<8} {d.stats.failed:<8} {exit_str}")
    elif len(status.dags) == 1:
        d = status.dags[0]
        state_str = "RUNNING" if d.is_running else (f"DONE (exit {d.exit_status})" if d.exit_status is not None else "PENDING")
        print(f"\nMaster DAG: {d.name} [{state_str}]")

    # 4. Error warnings if any
    if agg.failed > 0 or agg.held > 0 or status.active_errors:
        print("\nAlerts / Warnings:")
        if agg.held > 0:
            print(f"  [!] {agg.held} job(s) currently HELD in Condor queue.")
        if agg.failed > 0:
            print(f"  [!] {agg.failed} node(s) FAILED in DAGMan.")
        if status.active_errors:
            print(f"  [!] Non-empty error logs detected ({len(status.active_errors)} sample shown):")
            for ef in status.active_errors[:5]:
                print(f"      - {ef}")

    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# Email Notification
# -----------------------------------------------------------------------------

def send_email(
    to_email: str,
    subject: str,
    body: str,
    sender: str | None = None,
    smtp_server: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    smtp_tls: bool = False,
    smtp_ssl: bool = False,
    mail_cmd: str | None = None,
) -> tuple[bool, str]:
    """Sends notification email using the best available mechanism."""
    hostname = socket.gethostname()
    if not sender:
        username = os.environ.get("USER", "root")
        sender = f"{username}@{hostname}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg["Date"] = email.utils.formatdate(localtime=True)

    errors = []

    # 1. Custom mail command
    if mail_cmd:
        try:
            cmd_name = os.path.basename(mail_cmd)
            if cmd_name in ("mail", "mailx"):
                proc = subprocess.run([mail_cmd, "-s", subject, to_email], input=body, text=True, capture_output=True, timeout=15)
            elif "sendmail" in cmd_name:
                proc = subprocess.run([mail_cmd, "-t", "-oi"], input=msg.as_string(), text=True, capture_output=True, timeout=15)
            else:
                proc = subprocess.run([mail_cmd, to_email], input=body, text=True, capture_output=True, timeout=15)
            if proc.returncode == 0:
                return True, f"Sent via command '{mail_cmd}'"
            return False, f"Command '{mail_cmd}' failed: {proc.stderr.strip()}"
        except Exception as e:
            return False, f"Execution failed for '{mail_cmd}': {e}"

    # 2. Configured SMTP server
    if smtp_server:
        port = smtp_port or (465 if smtp_ssl else (587 if smtp_tls else 25))
        password = smtp_password or os.environ.get("SMTP_PASSWORD", "")
        try:
            server = smtplib.SMTP_SSL(smtp_server, port, timeout=15) if smtp_ssl else smtplib.SMTP(smtp_server, port, timeout=15)
            if smtp_tls and not smtp_ssl:
                server.starttls()
            if smtp_user:
                server.login(smtp_user, password)
            server.send_message(msg)
            server.quit()
            return True, f"Sent via SMTP server {smtp_server}:{port}"
        except Exception as e:
            return False, f"SMTP delivery failed: {e}"

    # 3. Local system mail tools (mailx / mail)
    for tool in ("mailx", "mail"):
        tool_path = shutil.which(tool)
        if tool_path:
            try:
                proc = subprocess.run([tool_path, "-s", subject, to_email], input=body, text=True, capture_output=True, timeout=15)
                if proc.returncode == 0:
                    return True, f"Sent via '{tool_path}'"
                errors.append(f"{tool} failed: {proc.stderr.strip()}")
            except Exception as e:
                errors.append(f"{tool} error: {e}")

    # 4. sendmail binary
    sendmail_path = shutil.which("sendmail") or "/usr/sbin/sendmail"
    if os.path.exists(sendmail_path):
        try:
            proc = subprocess.run([sendmail_path, "-t", "-oi"], input=msg.as_string(), text=True, capture_output=True, timeout=15)
            if proc.returncode == 0:
                return True, f"Sent via '{sendmail_path}'"
            errors.append(f"{sendmail_path} failed: {proc.stderr.strip()}")
        except Exception as e:
            errors.append(f"sendmail error: {e}")

    # 5. Localhost SMTP relay (127.0.0.1:25)
    try:
        server = smtplib.SMTP("127.0.0.1", 25, timeout=10)
        server.send_message(msg)
        server.quit()
        return True, "Sent via local SMTP relay (127.0.0.1:25)"
    except Exception as e:
        errors.append(f"Localhost SMTP failed: {e}")

    return False, "; ".join(errors) if errors else "No mail utility or SMTP server found."


def build_completion_report(status: PipelineStatus, start_time: datetime, check_count: int, interval_sec: float) -> str:
    """Builds a formatted email completion report."""
    now = datetime.now()
    duration_str = format_duration((now - start_time).total_seconds())
    hostname = socket.gethostname()
    agg = status.aggregate_stats

    header = "SUCCESS: ALL S-EPD CALIBRATION RUNS COMPLETED" if status.is_success else "ALERT: S-EPD CALIBRATION WORKFLOW FINISHED WITH ISSUES"

    report = f"""Hello,

This is an automated notification from {hostname}.

Status: {header}
Target Directory: {status.output_dir}

Execution Summary:
----------------------------------------------------------------------
Host:                {hostname}
Project Directory:   {status.output_dir}
DAG Directory:       {status.dag_dir}
Finalized Runs:      {len(status.finalized_runs)} / {status.total_runs} ({(len(status.finalized_runs) / status.total_runs * 100.0 if status.total_runs > 0 else 0.0):.1f}%)
Total DAG Nodes:     {agg.total}
Completed Nodes:     {agg.done}
Failed Nodes:        {agg.failed}
Held Jobs:           {agg.held}
Monitoring Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Finished At:         {now.strftime('%Y-%m-%d %H:%M:%S')}
Total Wait Time:     {duration_str}
Checks Performed:    {check_count}

DAGMan Partitions:
----------------------------------------------------------------------
"""
    for d in status.dags:
        state = "DONE (0)" if d.exit_status == 0 else f"EXIT({d.exit_status})"
        report += f"  - {d.name:<26}: {state:<10} (Done: {d.stats.done}/{d.stats.total}, Failed: {d.stats.failed})\n"

    if status.finalized_runs:
        sample_runs = status.finalized_runs[:20]
        report += f"\nSample Finalized Runs ({len(sample_runs)} of {len(status.finalized_runs)} shown):\n"
        report += f"  {', '.join(sample_runs)}\n"

    if status.pending_runs:
        report += f"\nUnfinished / Pending Runs ({len(status.pending_runs)}):\n"
        report += f"  {', '.join(status.pending_runs[:20])}\n"

    report += """\n----------------------------------------------------------------------
Output Deliverables:
  - QA Merged ROOT Files:     <output_dir>/QA/QA-<run>.root
  - Final Q-Vector Calib:     <output_dir>/QVecCalib/QVecCalib-<run>.root
  - CDB Calibration Payloads: <output_dir>/CDB/<run>/
----------------------------------------------------------------------
Notification sent by scripts/monitor_dag.py
"""
    return report


# -----------------------------------------------------------------------------
# Main Loop & CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor sEPD Calibration HTCondor DAGMan workflows until completion and notify via email.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional arguments
    parser.add_argument("pos_dir", nargs="?", default=None, metavar="DIR", help="Target project output directory or dag/ directory")
    parser.add_argument("pos_interval", nargs="?", default=None, metavar="INTERVAL", help="Polling interval, e.g. 30s, 1m, 5m (default: 60s)")
    parser.add_argument("pos_email", nargs="?", default=None, metavar="EMAIL", help="Recipient email address")

    # Flag options
    parser.add_argument("-d", "--dir", dest="flag_dir", default=None, help="Target project output directory")
    parser.add_argument("-i", "--interval", dest="flag_interval", default=None, help="Polling interval (default: 60s)")
    parser.add_argument("-e", "--email", dest="flag_email", default=None, help="Recipient email address")
    parser.add_argument("-s", "--subject", default=None, help="Custom email subject")
    parser.add_argument("--run-once", action="store_true", help="Inspect status once and exit (exit code 0 if all done, 1 if in-progress or failed)")
    parser.add_argument("--test-email", action="store_true", help="Send a test email immediately to verify mail delivery, then exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode: suppress periodic dashboard redraws")

    mail_grp = parser.add_argument_group("Email / SMTP Configuration Options")
    mail_grp.add_argument("--mail-cmd", default=None, help="Force specific mail binary (e.g. mailx, mail, /usr/sbin/sendmail)")
    mail_grp.add_argument("--smtp-server", default=None, help="SMTP server host (e.g. localhost, smtp.bnl.gov)")
    mail_grp.add_argument("--smtp-port", type=int, default=None, help="SMTP server port")
    mail_grp.add_argument("--smtp-user", default=None, help="SMTP username")
    mail_grp.add_argument("--smtp-password", default=None, help="SMTP password")
    mail_grp.add_argument("--smtp-tls", action="store_true", help="Use TLS for SMTP")
    mail_grp.add_argument("--smtp-ssl", action="store_true", help="Use SSL for SMTP")
    mail_grp.add_argument("--sender", default=None, help="Sender email address")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    raw_dir = args.flag_dir or args.pos_dir
    raw_interval = args.flag_interval or args.pos_interval or "60s"
    email_addr = args.flag_email or args.pos_email

    mail_kwargs = {
        "sender": args.sender,
        "smtp_server": args.smtp_server,
        "smtp_port": args.smtp_port,
        "smtp_user": args.smtp_user,
        "smtp_password": args.smtp_password,
        "smtp_tls": args.smtp_tls,
        "smtp_ssl": args.smtp_ssl,
        "mail_cmd": args.mail_cmd,
    }

    # Handle test-email mode
    if args.test_email:
        if not email_addr:
            parser.error("Recipient email is required for --test-email.")
        print(f"Sending test notification email to {email_addr}...")
        test_sub = f"[Test] sEPD Calibration Monitor Test from {socket.gethostname()}"
        test_body = f"Hello,\n\nTest notification from monitor_dag.py on {socket.gethostname()}.\nTimestamp: {datetime.now()}\n"
        ok, msg = send_email(to_email=email_addr, subject=test_sub, body=test_body, **mail_kwargs)
        if ok:
            print(f"[SUCCESS] {msg}")
            return 0
        print(f"[FAILED] {msg}", file=sys.stderr)
        return 1

    if not raw_dir:
        parser.error("Target project directory is required. Specify via argument or -d/--dir.")

    try:
        interval_sec = parse_interval(raw_interval)
    except ValueError as e:
        parser.error(str(e))

    target_dir = Path(raw_dir).resolve()
    start_time = datetime.now()
    check_count = 0

    # Initial check
    status = inspect_pipeline(target_dir)

    if args.run_once:
        print_dashboard(status, start_time=start_time, check_count=1, interval_sec=interval_sec)
        return 0 if status.is_success else 1

    print(f"Starting DAG completion monitor on: {target_dir}")
    print(f"Polling interval: {interval_sec:.0f}s | Notification email: {email_addr or 'None'}")
    if email_addr:
        print(f"Notification will be dispatched automatically once all DAGMan processes finish.\n")

    while True:
        check_count += 1
        status = inspect_pipeline(target_dir)

        if not args.quiet:
            print_dashboard(status, start_time=start_time, check_count=check_count, interval_sec=interval_sec)

        # Completion check
        if status.is_all_complete:
            print("\n" + "=" * 78)
            print(" [WORKFLOW COMPLETED] All DAGMan processes have exited!")
            print("=" * 78)

            if email_addr:
                subject = args.subject or (
                    f"[{('SUCCESS' if status.is_success else 'ALERT')}] "
                    f"sEPD Calibration Workflow Complete: {len(status.finalized_runs)}/{status.total_runs} runs on {socket.gethostname()}"
                )
                report_body = build_completion_report(status, start_time, check_count, interval_sec)
                print(f"Sending completion report email to {email_addr}...")
                ok, msg = send_email(to_email=email_addr, subject=subject, body=report_body, **mail_kwargs)
                if ok:
                    print(f"[SUCCESS] Email sent: {msg}")
                else:
                    print(f"[WARNING] Email delivery failed: {msg}", file=sys.stderr)
            break

        time.sleep(interval_sec)

    return 0 if status.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
