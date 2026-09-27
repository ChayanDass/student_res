"""Shared logging, system-info, step-tracking, and RAM-budget helpers.

Every stage print in train.py / predict.py / features.py / blocking.py goes
through ``log`` so output is timestamped and explicitly flushed. This matters
because Python's stdout is fully buffered (not line-buffered) once it is
redirected to a file — e.g. ``nohup python train.py > train.log 2>&1 &`` —
so a plain ``print`` can sit invisible in an in-process buffer for hours
until the buffer fills or the process exits. That silent buffering, not an
actual crash, was the root cause of a training run that looked "stuck" with
an empty log file. ``log`` always flushes, so every line lands in the file
the moment it is printed regardless of redirection.

Run with ``python -u`` as a second, belt-and-suspenders layer of protection
(forces the interpreter itself into fully unbuffered mode).

This module also centralizes the "use most of the box, but leave headroom"
policy for the AWS instance this pipeline runs on: ``ram_budget_gb()`` caps
memory-driven sizing decisions (worker count, chunk size) at a fraction
(default 70%) of total system RAM, so the OOM killer never gets a chance to
land mid-run on an unattended multi-hour job.
"""
import os
import platform
import time

_START = time.time()

# Fraction of total system RAM the pipeline is allowed to plan around. Kept
# well under 100% so the OS, page cache, and other processes always have
# headroom — an OOM kill mid-training is far more expensive than the
# throughput left on the table by not using the last 30% of RAM.
RAM_FRACTION = float(os.environ.get("BER_RAM_FRACTION", "0.7"))


def log(msg: str) -> None:
    elapsed = time.time() - _START
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} +{elapsed:8.1f}s] {msg}", flush=True)


def _meminfo() -> dict:
    """Raw /proc/meminfo values in kB. Empty dict off Linux (e.g. local macOS)."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":", 1)
                info[key] = int(val.strip().split()[0])
    except Exception:
        pass
    return info


def mem_total_gb() -> float:
    info = _meminfo()
    if "MemTotal" in info:
        return info["MemTotal"] / (1024 ** 2)
    return 0.0


def mem_available_gb() -> float:
    info = _meminfo()
    if info:
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return avail / (1024 ** 2)
    return 0.0


def set_ram_fraction(fraction: float) -> None:
    """Override the module-wide RAM_FRACTION policy (e.g. from a --mem-fraction
    CLI flag). Affects every subsequent ram_budget_gb()/auto_n_jobs()/
    scale_to_ram() call that doesn't pass its own explicit ``fraction``."""
    global RAM_FRACTION
    RAM_FRACTION = fraction


def ram_budget_gb(fraction: float = None) -> float:
    """Total RAM the pipeline plans around: ``fraction`` (default
    ``RAM_FRACTION`` = 70%) of total system RAM. Used to size worker counts
    and streaming-chunk row counts so a full-scale run stays well clear of
    the OOM killer regardless of what else is on the box."""
    fraction = RAM_FRACTION if fraction is None else fraction
    total = mem_total_gb()
    if total <= 0:
        return 4.0  # conservative fallback when /proc/meminfo is unavailable
    return total * fraction


def cpu_count() -> int:
    return os.cpu_count() or 1


def auto_n_jobs(mem_per_worker_gb: float = 0.5, fraction: float = None) -> int:
    """Worker count for a multiprocessing.Pool that respects BOTH available
    cores and the RAM budget: ``ram_budget_gb() / mem_per_worker_gb`` caps the
    core-count answer so a wide, RAM-hungry pool can't push the box past the
    70% ceiling. Always returns at least 1."""
    by_cpu = cpu_count()
    budget = ram_budget_gb(fraction)
    by_mem = max(1, int(budget / mem_per_worker_gb)) if mem_per_worker_gb > 0 else by_cpu
    return max(1, min(by_cpu, by_mem))


# The hand-tuned batch-size defaults below (100k / 300k / 200k rows) were
# sized for an m6i.2xlarge (32 GB RAM) running at the default 70% budget,
# i.e. a ~22.4 GB working set. ``scale_to_ram`` scales those references
# proportionally to whatever box is actually running the job, so a bigger
# (or smaller) instance automatically gets a bigger (or smaller) batch size
# instead of silently under- or over-using the RAM that's actually there.
REFERENCE_RAM_BUDGET_GB = 32.0 * 0.7


def scale_to_ram(reference_rows: int, fraction: float = None, min_rows: int = 10_000,
                  max_rows: int = 5_000_000) -> int:
    budget = ram_budget_gb(fraction)
    scale = budget / REFERENCE_RAM_BUDGET_GB if REFERENCE_RAM_BUDGET_GB > 0 else 1.0
    rows = int(reference_rows * scale)
    return max(min_rows, min(max_rows, rows))


def mem_snapshot() -> str:
    """Return 'used/total GB (pct%)' read from /proc/meminfo (Linux only;
    returns 'n/a' anywhere that file doesn't exist, e.g. local macOS dev)."""
    total = mem_total_gb()
    if total <= 0:
        return "n/a"
    avail = mem_available_gb()
    used = total - avail
    pct = used / total * 100 if total else 0.0
    return f"{used:.1f}/{total:.1f}GB ({pct:.0f}%)"


def load_snapshot() -> str:
    try:
        l1, l5, _l15 = os.getloadavg()
        return f"load {l1:.1f}/{l5:.1f} ({cpu_count()} cores)"
    except Exception:
        return "n/a"


def resource_line() -> str:
    return f"mem={mem_snapshot()} {load_snapshot()}"


def disk_free_gb(path: str = ".") -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / (1024 ** 3)
    except Exception:
        return 0.0


def system_info_banner(extra: dict = None) -> None:
    """Print a one-time banner with everything needed to sanity-check the
    machine this run landed on: CPU, RAM, RAM budget (70% policy), disk, OS,
    Python/host. ``extra`` adds run-specific key/values (e.g. resolved n_jobs,
    batch sizes) to the same banner."""
    total = mem_total_gb()
    budget = ram_budget_gb()
    lines = [
        "===== SYSTEM INFO =====",
        f"  host        : {platform.node()}",
        f"  platform    : {platform.system()} {platform.release()} ({platform.machine()})",
        f"  python      : {platform.python_version()}",
        f"  cpu cores   : {cpu_count()}",
        f"  ram total   : {total:.1f} GB" if total else "  ram total   : n/a",
        f"  ram budget  : {budget:.1f} GB ({RAM_FRACTION * 100:.0f}% cap)" if total else "  ram budget  : n/a",
        f"  disk free   : {disk_free_gb():.1f} GB",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"  {k:<12}: {v}")
    lines.append("=" * 23)
    for line in lines:
        log(line)


class Steps:
    """Numbered step tracker so long-running stages print 'STEP i/N: ...'
    instead of an unlabelled wall of log lines — makes it obvious from the
    log alone how far through the pipeline a run is."""

    def __init__(self, total: int):
        self.total = total
        self.current = 0

    def start(self, label: str) -> None:
        self.current += 1
        log_stage(f"STEP {self.current}/{self.total}: {label}")


class ProgressBar:
    """Textual progress bar for a chunked/batched loop: logs current process
    name, entries done/total, throughput, ETA, and live resource usage —
    every long-running loop in the pipeline should report through this
    instead of running silently between a start and a final done message."""

    def __init__(self, total: int, label: str, report_every_pct: float = 10.0):
        self.total = total
        self.label = label
        self.done = 0
        self.t0 = time.time()
        self._report_every = max(1, int(total * report_every_pct / 100)) if total else 1
        self._last_reported = 0
        if total:
            log(f"  [{label}] starting: {total:,} total")
        else:
            log(f"  [{label}] starting (0 total — nothing to do)")

    def update(self, n: int = 1) -> None:
        self.done += n
        if self.done - self._last_reported >= self._report_every or self.done >= self.total:
            self._report()
            self._last_reported = self.done

    def _report(self) -> None:
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        pct = (self.done / self.total * 100) if self.total else 100.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        log(f"  [{self.label}] {self.done:,}/{self.total:,} ({pct:.0f}%)  "
            f"{rate:,.0f}/s  eta={eta:.0f}s  {resource_line()}")

    def close(self) -> None:
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        log(f"  [{self.label}] done: {self.done:,}/{self.total:,} in {elapsed:.1f}s "
            f"({rate:,.0f}/s)  {resource_line()}")


def log_stage(stage: str) -> None:
    log(f"===== {stage} | {resource_line()} =====")
