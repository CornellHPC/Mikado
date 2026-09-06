"""Shared run-record helper for the example benchmarks.

Each benchmark script accepts ``--record <file>`` and appends one JSON object
per run (JSON Lines). Appends are no-read and guarded by a best-effort
exclusive ``flock`` so parallel runs can safely share a record file, and each
line is independently parseable (``pandas.read_json(lines=True)``, ``jq``, etc.).
"""

import fcntl
import functools
import json
import os
import pathlib
import socket
import subprocess
from datetime import datetime


def _cpu_model():
    """First ``model name`` line from ``/proc/cpuinfo`` (None if unavailable)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _cpu_counts():
    """Return ``(physical_cores, logical_threads)``.

    Physical cores are counted as distinct ``(physical id, core id)`` pairs in
    ``/proc/cpuinfo``; ``cores`` is None on platforms that omit those fields.
    """
    threads = os.cpu_count()
    cores = None
    try:
        seen = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        seen.add((phys, core))
                    phys = core = None
        cores = len(seen) or None
    except OSError:
        pass
    return cores, threads


def _gpu_info():
    """Return ``(gpu_model, gpu_count)`` via ``nvidia-smi`` (``(None, 0)`` if absent)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None, 0
    if out.returncode != 0:
        return None, 0
    names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not names:
        return None, 0
    return names[0], len(names)


def _ram_gib():
    """Total RAM in GiB (rounded to 1 decimal) from ``/proc/meminfo``."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _numa_binding():
    """Process NUMA binding (``policy``/``cpubind``/``nodebind``/``membind``).

    Prefers ``numactl --show``: it reports the active memory policy set by
    ``numactl --membind`` (via ``set_mempolicy``), which ``/proc/self/status``
    ``Mems_allowed_list`` does *not* reflect. ``policy: default`` with all nodes
    listed means no binding is in effect. Falls back to the cpuset allowed lists
    from ``/proc/self/status`` if ``numactl`` is unavailable. Returns None if
    nothing can be determined.
    """
    try:
        out = subprocess.run(
            ["numactl", "--show"], capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            info = {}
            for line in out.stdout.splitlines():
                key, _, val = line.partition(":")
                key = key.strip()
                if key in ("policy", "cpubind", "nodebind", "membind"):
                    info[key] = val.strip()
            if info:
                return info
    except (OSError, subprocess.SubprocessError):
        pass

    # Fallback: cpuset allowed masks (captures --cpunodebind, not --membind).
    info = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Cpus_allowed_list:"):
                    info["cpus_allowed"] = line.split(":", 1)[1].strip()
                elif line.startswith("Mems_allowed_list:"):
                    info["mems_allowed"] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return info or None


@functools.lru_cache(maxsize=1)
def machine_info():
    """Best-effort hardware/binding description for the current node.

    Cached so the probes (notably the ``nvidia-smi`` subprocess) run once per
    process even when several records are written. Never raises; any failed
    probe degrades to ``None``/``0``.
    """
    try:
        cores, threads = _cpu_counts()
        gpu_model, gpu_count = _gpu_info()
        return {
            "hostname": socket.gethostname(),
            "cpu_model": _cpu_model(),
            "cpu_cores": cores,
            "cpu_threads": threads,
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "ram_gib": _ram_gib(),
            "numa": _numa_binding(),
        }
    except Exception:
        return None


def _format_size(num_bytes):
    """Human-readable size string: ``"X.XX GB"`` if >= 1 GiB else ``"X.XX MB"``.

    Returns None for non-numeric/negative input rather than raising.
    """
    try:
        num_bytes = int(num_bytes)
    except (TypeError, ValueError):
        return None
    if num_bytes < 0:
        return None
    gib = 1024 ** 3
    mib = 1024 ** 2
    if num_bytes >= gib:
        return f"{num_bytes / gib:.2f} GB"
    return f"{num_bytes / mib:.2f} MB"


def input_files_summary(paths):
    """Summarize the input files actually used: ``[{"name", "size"}, ...]``.

    For each path, records the bare filename (name + extension) and a
    human-readable size. Best-effort per file: a missing or unstattable path
    still yields an entry with ``size: None`` rather than failing the run.
    """
    summary = []
    for p in paths:
        p = pathlib.Path(p)
        try:
            size = _format_size(p.stat().st_size)
        except OSError:
            size = None
        summary.append({"name": p.name, "size": size})
    return summary


def _jsonify(value):
    """Coerce arbitrary values into JSON-serializable form.

    Handles the types that show up in argparse namespaces (``pathlib.Path``,
    nested dicts/lists, tuples) and falls back to ``str`` for anything else.
    """
    if isinstance(value, pathlib.PurePath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_record(path, *, script, application, params, metrics,
                 output_summary=None, input_files=None):
    """Append one JSON-Lines run record to ``path`` (created if missing).

    ``application`` is a top-level tag ("bolt" | "pca" | "pca_lobpcg" | "gwas" |
    "kernel") identifying the analysis, since the same script name (e.g.
    ``grapp_cusparse.py``) exists under several of them.
    ``input_files`` is the list of files actually used as input (see
    :func:`input_files_summary`). The remaining fields are written in a fixed
    order: script executed, execution time, machine description, input
    parameters, input files, output summary, and performance metrics.
    """
    path = pathlib.Path(path)
    record = {
        "application": application,
        "script": script,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": _jsonify(machine_info()),
        "params": _jsonify(params),
        "input_files": _jsonify(input_files),
        "output_summary": _jsonify(output_summary),
        "metrics": _jsonify(metrics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        # flock is best-effort: some parallel filesystems (Lustre/GPFS) reject it
        # with ENOTSUPP/EOPNOTSUPP. A single-line append is already effectively
        # atomic, so fall back to an unlocked write rather than failing the run.
        locked = False
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            locked = True
        except OSError:
            pass
        try:
            fh.write(line)
        finally:
            if locked:
                fcntl.flock(fh, fcntl.LOCK_UN)
