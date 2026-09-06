#!/usr/bin/env python3
"""Single endpoint for running the benchmark scripts under ``examples/``.

Picks an application + backend + dataset, optionally pins the whole process to
one CPU socket, runs a configurable number of warmup (unrecorded) and measured
(recorded) iterations, and forwards any extra arguments to the underlying
benchmark script. Each benchmark already appends one JSON-Lines record per run
via its ``--record`` flag (see ``examples/_record.py``); this launcher only
passes ``--record`` on measured runs and reads those records back to print an
aggregate timing summary.

The cusparse backend's knobs are first-class launcher params: ``--device-map``
(required for cusparse), and ``--capture``/``--native`` which default to on and
can be turned off with ``--no-capture``/``--no-native``. The mkl backend's knobs
are likewise first-class: ``--mkl-threads`` (int per GRG or a JSON thread-map
file) and ``--optimize``.

Examples
--------
    python evaluate.py -a bolt -b cusparse \
        -d /path/to/grg_spmv_8_filtered_maf0.01/ \
        --socket 0 --warmup 1 --runs 3 \
        --device-map configs/device_map_4.json \
        --record records/run.jsonl -- --skip-output

    python evaluate.py -a pca -b mkl \
        -d /path/to/grg_spmv_8_filtered_maf0.01/ \
        --socket 0 --runs 3 --mkl-threads 8 \
        --record records/pca_mkl.jsonl -- --skip-output

    python evaluate.py -a bolt -b boltlmm \
        -d /path/to/plink_filtered_maf0.01 --runs 1 --record records/bolt.jsonl \
        --bolt-pheno-file pheno.bolt.phen \
        --work-dir /tmp/work --output records/bolt_results.tsv \
        -- --chromosomes 1,2,3 --bolt-bin /path/to/bolt

The optional ``--bolt-pheno-file`` (bolt application only) is forwarded as
``--pheno-file`` to whichever bolt backend is selected; without it each backend
falls back to its own synthetic seed-generated phenotype. For the boltlmm/official
BOLT backend the file must be in ``FID IID PHENO`` form; the grapp backends read
its last column.

The ``pca_lobpcg`` application is the LOBPCG-solver counterpart of ``pca`` (same
cusparse/grgl/mkl backends and flags); its LOBPCG-only knobs (``--pcs``,
``--maxiter``, ``--seed``, ``--tol``) are forwarded after ``--``:

    python evaluate.py -a pca_lobpcg -b cusparse \
        -d /path/to/grg_spmv_8_filtered_maf0.01/ \
        --device-map configs/device_map_4.json \
        --runs 3 --record records/pca_lobpcg_cusparse.jsonl \
        -- --pcs 10 --maxiter 100 --skip-output

The ``gwas`` application runs a basic per-SNP linear-regression GWAS (the grgl/mkl/
cusparse grapp backends plus the official plink2 ``--glm`` baseline). It is single
fileset: the dataset is one ``.grg`` (grgl), ``.grg_spmv`` (mkl/cusparse), or PLINK
``.bed``/``.pgen`` path or stem (plink), forwarded positionally.
Phenotype/covariates and the per-backend knobs are forwarded after ``--`` (e.g.
``-p pheno.txt -c covars.txt -b -s``; omit ``-p`` for a synthetic phenotype).
cusparse runs copy-mode only: ``--native`` is always disabled for gwas (grapp
drives the GRG through its numpy operators), but ``--device-map`` is still
required and ``--capture``/``--force-spmm`` apply:

    python evaluate.py -a gwas -b cusparse \
        -d /path/to/chr1.grg_spmv \
        --device-map configs/device_map_1.json \
        --runs 3 --record records/gwas_cusparse.jsonl -- -p pheno.txt --skip-output

    python evaluate.py -a gwas -b mkl \
        -d /path/to/chr1.grg_spmv --mkl-threads 8 \
        --runs 3 --record records/gwas_mkl.jsonl -- --skip-output

The ``kernel`` application times only the matmul kernel (no algorithm). Its
per-run knobs (``--direction``, ``--k``, ``--dtype``, ``--threads``, the script's
own ``--trials``/``--warmup``) are forwarded after ``--``:

    python evaluate.py -a kernel -b cusparse \
        -d /path/to/grg_spmv_8_filtered_maf0.01/ \
        --device-map configs/device_map_4.json \
        --runs 3 --record records/kernel.jsonl -- --direction up --k 16

The ``kernel``/``trsv`` backend instead runs a pure cuSPARSE SpTrSV on a single
``.csr`` triangular matrix (``M = I - A``, built offline by
``code/trsv/grg_to_csr.py``). The dataset is one ``.csr`` file, and it
is single-GPU (no ``--device-map``):

    python evaluate.py -a kernel -b trsv \
        -d /path/to/chr1.csr \
        --runs 3 --record records/kernel_trsv.jsonl -- --direction up --k 4
"""

import argparse
import json
import logging
import pathlib
import shutil
import statistics
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent
EXAMPLES_DIR = REPO_ROOT / "examples"

log = logging.getLogger("evaluate")

# cusparse-only boolean knobs, as (argparse dest, user-facing flag pair). They are
# parsed as tri-state (None = not given) so main() can tell an explicit toggle from
# the default and warn when one is passed to a backend that ignores it.
CUSPARSE_TOGGLES = (
    ("capture", "--capture/--no-capture"),
    ("native", "--native/--no-native"),
    ("force_spmm", "--force-spmm/--no-force-spmm"),
)

# Value each toggle takes when it is not given. These mirror the defaults in the
# cusparse scripts themselves, so running a script directly and running it through
# this launcher configure the same thing.
CUSPARSE_TOGGLE_DEFAULTS = {"capture": True, "native": True, "force_spmm": False}


def _sockets_to_nodes():
    """Map of ``{physical_socket: [numa_node_id, ...]}`` read from sysfs.

    A socket can span several NUMA nodes (e.g. AMD NPS4 has 4 nodes/socket), so
    binding to a socket means binding to all of its nodes. Robust across NPS
    configs because it reads the actual node->socket map from sysfs.
    """
    node_dir = pathlib.Path("/sys/devices/system/node")
    by_socket = {}
    for nd in sorted(node_dir.glob("node[0-9]*")):
        node = int(nd.name[len("node"):])
        cpulist = (nd / "cpulist").read_text().strip()
        if not cpulist:
            continue
        first_cpu = int(cpulist.split(",")[0].split("-")[0])
        pkg = int((pathlib.Path("/sys/devices/system/cpu")
                   / f"cpu{first_cpu}" / "topology" / "physical_package_id")
                  .read_text().strip())
        by_socket.setdefault(pkg, []).append(node)
    return by_socket


def num_sockets():
    """Number of distinct physical CPU sockets (via sysfs); at least 1."""
    return len(_sockets_to_nodes()) or 1


def system_memory_mb():
    """Total RAM in MB from ``/proc/meminfo`` (``MemTotal`` is reported in kB)."""
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    sys.exit("error: could not read MemTotal from /proc/meminfo")


def socket_numa_nodes(socket):
    """NUMA node ids belonging to the given physical socket (via sysfs)."""
    by_socket = _sockets_to_nodes()
    nodes = sorted(by_socket.get(socket, []))
    if not nodes:
        sys.exit(
            f"error: could not determine NUMA nodes for socket {socket}; "
            f"available sockets: {', '.join(str(s) for s in sorted(by_socket))}"
        )
    return nodes


# Dataset adapters translate the single ``--dataset`` value into the arg list
# each backend's script expects.
def _positional(dataset):
    """grapp backends take the dataset as a positional input path."""
    return [dataset]


def _plink_dir(dataset):
    """boltlmm takes the dataset directory via ``--plink-dir``."""
    return ["--plink-dir", dataset]


# Backends shared by every application (the grapp GPU/CPU paths).
_SHARED_BACKENDS = {
    "cusparse": ("grapp_cusparse.py", _positional),
    "grgl": ("grapp_grgl.py", _positional),
    "mkl": ("grapp_mkl.py", _positional),
}

# Kernel benchmarks live under examples/kernel/ with their own (non-grapp) script
# names. They time only the matmul kernel and expose a reduced cusparse/mkl flag
# surface (no --force-spmm/--tol-record on cusparse); see build_cmd.
_KERNEL_BACKENDS = {
    "cusparse": ("cusparse.py", _positional),
    "grgl": ("grgl.py", _positional),
    "mkl": ("mkl.py", _positional),
    # Pure cuSPARSE SpTrSV on a single .csr triangular matrix (M = I - A). The
    # dataset is one .csr file (built offline by code/trsv/grg_to_csr.py), not a
    # .grg_spmv directory. trsv.py wraps the code/trsv/sptrsv binary.
    "trsv": ("trsv.py", _positional),
    # Legacy GPUGRG matmul kernel via pygrgl.load_gpu_grg (no grapp). The dataset
    # is a directory of (or explicit list of) precomputed .gpugrg files, built
    # offline by grg_to_gpugrg.py. Single-GPU (no --device-map).
    "legacy": ("legacy_gpugrg.py", _positional),
}

# Per-application backend registry: shared grapp backends plus each
# application's own baseline/reference implementation. There is intentionally no
# global "original" backend -- the baseline differs by application.
BACKENDS = {
    "bolt": {
        **_SHARED_BACKENDS,
        "boltlmm": ("boltlmm.py", _plink_dir),
    },
    # The plink baseline takes the dataset directory via --plink-dir and picks up
    # either chr<N>.{bim,bed,fam} or chr<N>.{pvar,pgen,psam} from it.
    "pca": {
        **_SHARED_BACKENDS,
        "plink": ("plink.py", _plink_dir),
    },
    # Basic per-SNP GWAS. grapp backends (grgl/mkl/cusparse) plus the official
    # plink2 --glm baseline. Single fileset: the dataset is one .grg (grgl),
    # .grg_spmv (mkl/cusparse), or a PLINK .bed/.pgen path or stem (plink),
    # forwarded positionally.
    # Phenotype/covariates are passed after -- (e.g. -- -p pheno.txt -c covars.txt
    # for grapp; -- --plink2-bin ... for plink). See the cusparse device-map/native
    # note below.
    "gwas": {
        **_SHARED_BACKENDS,
        "plink": ("plink.py", _positional),
    },
    # LOBPCG-solver counterparts of the eigsh pca backends. Backend names are kept
    # identical to "pca" so build_cmd's cusparse/mkl flag dispatch and main()'s
    # validation apply unchanged; only the script (and examples/ subdir) differs.
    # No plink baseline -- LOBPCG is a grapp-only path.
    "pca_lobpcg": {
        "cusparse": ("grapp_cusparse_lobpcg.py", _positional),
        "grgl": ("grapp_grgl_lobpcg.py", _positional),
        "mkl": ("grapp_mkl_lobpcg.py", _positional),
    },
    "kernel": _KERNEL_BACKENDS,
}


def resolve_backend(application, backend):
    """Return ``(script_path, dataset_adapter)`` or exit with a clear error."""
    app_backends = BACKENDS.get(application)
    if app_backends is None:
        sys.exit(
            f"error: unknown application '{application}'. "
            f"Known applications: {', '.join(sorted(BACKENDS))}"
        )
    entry = app_backends.get(backend)
    if entry is None:
        sys.exit(
            f"error: backend '{backend}' is not registered for application "
            f"'{application}'. Available backends: {', '.join(sorted(app_backends))}"
        )
    script_name, adapter = entry
    script_path = EXAMPLES_DIR / application / script_name
    if not script_path.is_file():
        sys.exit(f"error: benchmark script not found: {script_path}")
    log.debug("resolved %s/%s -> %s", application, backend, script_path)
    return script_path, adapter


def build_cmd(args, script_path, adapter, record):
    """Assemble the full argv for one invocation of the benchmark script."""
    cmd = []
    if args.socket is not None:
        numactl = shutil.which("numactl")
        if numactl is None:
            sys.exit("error: --socket requested but 'numactl' was not found on PATH")
        nodes = ",".join(str(n) for n in socket_numa_nodes(args.socket))
        log.debug("socket %s -> NUMA nodes %s", args.socket, nodes)
        cmd += [numactl, f"--cpunodebind={nodes}", f"--membind={nodes}"]
    cmd += [args.python, str(script_path)]
    cmd += adapter(args.dataset)
    if args.backend == "cusparse":
        if args.capture:
            cmd.append("--capture")
        if args.native:
            cmd.append("--native")
        cmd += ["--device-map", str(args.device_map)]
        # The kernel cusparse script has no --force-spmm/--tol-record knobs (its
        # SpMV-vs-SpMM path is driven by k); only the grapp scripts take them.
        if args.application != "kernel":
            cmd.append("--force-spmm" if args.force_spmm else "--no-force-spmm")
            if args.tol_record is not None:
                cmd += ["--tol-record", str(args.tol_record)]
    if args.backend == "mkl":
        if args.mkl_threads is not None:
            cmd += ["--mkl-threads", str(args.mkl_threads)]
        if args.optimize:
            cmd.append("--optimize")
    if args.backend == "plink" and args.socket is not None:
        # --socket pins the run to one socket via numactl, so plink2 may only use
        # that socket's share of RAM. Cap its --memory accordingly (MB).
        sockets_allowed = 1  # --socket binds exactly one socket
        mem_mb = int(system_memory_mb() / num_sockets()
                     * sockets_allowed * args.plink_mem_fraction)
        log.debug("plink --memory cap: %d MB", mem_mb)
        cmd += ["--memory", str(mem_mb)]
    if args.bolt_pheno_file is not None:
        # Every bolt backend (boltlmm + the three grapp scripts) takes --pheno-file.
        cmd += ["--pheno-file", str(args.bolt_pheno_file)]
    if args.output is not None:
        # Every backend except the kernel scripts takes --output (a result TSV path).
        cmd += ["--output", str(args.output)]
    if args.work_dir is not None:
        # Only the boltlmm and plink baselines write intermediates to a --work-dir.
        cmd += ["--work-dir", str(args.work_dir)]
    if args.skip_output:
        # All grapp backends + the plink baseline accept --skip-output.
        cmd.append("--skip-output")
    cmd += args.extra
    if record and args.record is not None:
        cmd += ["--record", str(args.record)]
    return cmd


def _line_count(path):
    """Number of lines currently in ``path`` (0 if it does not exist)."""
    path = pathlib.Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def print_summary(record_path, start_line):
    """Read the records appended during this run and print mean/min/max metrics."""
    path = pathlib.Path(record_path)
    if not path.exists():
        print("No record file was written; skipping summary.")
        return
    with path.open("r", encoding="utf-8") as fh:
        new_lines = fh.readlines()[start_line:]

    metrics = {}  # key -> list of values across measured runs
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in (record.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(value)

    if not metrics:
        print("No structured metrics found in the new records; skipping summary.")
        return

    n = max(len(v) for v in metrics.values())
    print()
    print(f"=== Aggregate over {n} measured run(s) ===")
    width = max(len(k) for k in metrics)
    print(f"{'metric'.ljust(width)}   {'mean':>12}   {'min':>12}   {'max':>12}")
    for key in sorted(metrics):
        vals = metrics[key]
        mean = statistics.mean(vals)
        print(f"{key.ljust(width)}   {mean:12.4f}   {min(vals):12.4f}   {max(vals):12.4f}")


def run_iteration(label, cmd):
    """Echo and run one iteration, exiting on a non-zero child exit code."""
    print(f"\n[{label}] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"error: [{label}] exited with status {result.returncode}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-a", "--application", required=True,
                    help=f"benchmark application (one of: {', '.join(sorted(BACKENDS))})")
    ap.add_argument("-b", "--backend", required=True,
                    help="backend implementation (validated against the chosen application)")
    ap.add_argument("-d", "--dataset", required=True,
                    help="dataset path, forwarded to the backend via its dataset adapter")
    ap.add_argument("--python", default="python3",
                    help="python interpreter used to run the benchmark script (default: python3)")
    ap.add_argument("--socket", type=int, default=None,
                    help="bind the whole process to this CPU socket via numactl "
                         "(binds to all NUMA nodes of the socket); no binding if omitted")
    ap.add_argument("--plink-mem-fraction", dest="plink_mem_fraction", type=float,
                    default=0.8,
                    help="plink only: fraction of a socket's share of system RAM to "
                         "pass as plink2 --memory when --socket is set (default: 0.8)")
    ap.add_argument("--device-map", dest="device_map", type=pathlib.Path, default=None,
                    help="JSON device map forwarded to the cusparse backend "
                         "(required for -b cusparse; not used by other backends)")
    # These three default to None rather than True so that "not given" is
    # distinguishable from an explicit "--capture"; main() normalizes None to the
    # documented default (on) for cusparse and warns about it for every other
    # backend. See CUSPARSE_TOGGLES.
    ap.add_argument("--capture", action=argparse.BooleanOptionalAction, default=None,
                    help="cusparse only: enable CUDA graph capture (default: on; "
                         "disable with --no-capture)")
    ap.add_argument("--native", action=argparse.BooleanOptionalAction, default=None,
                    help="cusparse only: enable GPU-native I/O, requires capture "
                         "(default: on; disable with --no-native)")
    ap.add_argument("--force-spmm", dest="force_spmm",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="cusparse only: capture graphs at k=2 (SpMM) instead of "
                         "k=1 (SpMV) (default: off; enable with --force-spmm)")
    ap.add_argument("--tol-record", dest="tol_record", type=pathlib.Path, default=None,
                    help="cusparse native only: record JSONL (e.g. an MKL run record) "
                         "forwarded to the backend, which looks up the matching case's "
                         "effective_tol and uses it as --tol (see grapp_cusparse.py)")
    ap.add_argument("--mkl-threads", dest="mkl_threads", default=None,
                    help="mkl only: MKL SpMV threads per GRG, forwarded to the mkl "
                         "backend. Either an int (same count for every GRG) or a path "
                         "to a JSON thread-map file. Omitted -> backend default (1/GRG)")
    ap.add_argument("--optimize", action="store_true",
                    help="mkl only: enable MKL inspector-executor optimization")
    ap.add_argument("--bolt-pheno-file", dest="bolt_pheno_file", type=pathlib.Path,
                    default=None,
                    help="bolt only: phenotype file forwarded as --pheno-file to every "
                         "bolt backend. For boltlmm/official BOLT it must be a "
                         "'FID IID PHENO' file; the grapp backends read its last "
                         "column. Omitted -> each backend's "
                         "synthetic seed-generated phenotype")
    ap.add_argument("--output", type=pathlib.Path, default=None,
                    help="result TSV path forwarded as --output to the backend (every "
                         "application except kernel, which has no --output). With --runs "
                         ">1 each run overwrites it. Omitted -> the backend's default "
                         "filename in the CWD")
    ap.add_argument("--work-dir", dest="work_dir", type=pathlib.Path, default=None,
                    help="intermediate-files directory forwarded as --work-dir; only the "
                         "boltlmm and plink baselines accept it (both require one, so set "
                         "it here or after --)")
    ap.add_argument("--skip-output", dest="skip_output", action="store_true",
                    help="forward --skip-output to skip writing the result TSV (timings/"
                         "record still run). Supported by every backend except boltlmm "
                         "(always writes its stats) and the kernel scripts (write no TSV)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="number of unrecorded warmup runs (default: 0)")
    ap.add_argument("--runs", type=int, default=1,
                    help="number of measured (recorded) runs (default: 1)")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="JSON-Lines record file; passed only to measured runs")
    ap.add_argument("--debug", action="store_true",
                    help="set the log level to DEBUG (default: INFO)")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args forwarded to the benchmark script (use '--' to separate)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # argparse.REMAINDER keeps a leading "--" separator; drop it for cleanliness.
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    if args.warmup < 0 or args.runs < 1:
        sys.exit("error: --warmup must be >= 0 and --runs must be >= 1")

    if args.backend == "cusparse":
        if args.device_map is None:
            sys.exit("error: backend 'cusparse' requires --device-map")
        # Resolve the tri-state toggles to their documented defaults.
        for attr, _flag in CUSPARSE_TOGGLES:
            if getattr(args, attr) is None:
                setattr(args, attr, CUSPARSE_TOGGLE_DEFAULTS[attr])
        if args.application == "gwas" and args.native:
            # gwas drives the GRG through grapp's numpy SciPy operators (copy mode);
            # GPU-native I/O is not supported, so never forward --native for gwas.
            args.native = False
            print("warning: --native disabled for the gwas application "
                  "(copy-mode only)", file=sys.stderr)
        if not args.capture and args.native:
            args.native = False
            print("warning: --native disabled because capture is off "
                  "(native requires capture)", file=sys.stderr)
    else:
        # build_cmd only reads these for cusparse, so they would otherwise be
        # dropped without a word. Warn rather than exit: unlike --device-map they
        # have a meaningful default, so an ignored one does not change the run.
        given = [flag for attr, flag in CUSPARSE_TOGGLES
                 if getattr(args, attr) is not None]
        if given:
            print(f"warning: {', '.join(given)} "
                  f"{'is' if len(given) == 1 else 'are'} only used by the cusparse "
                  f"backend; ignored for '{args.backend}'", file=sys.stderr)
        if args.device_map is not None:
            sys.exit(
                f"error: --device-map is only used by the cusparse backend, "
                f"not '{args.backend}'"
            )

    if args.backend != "mkl" and (args.mkl_threads is not None or args.optimize):
        sys.exit(
            f"error: --mkl-threads/--optimize are only used by the mkl backend, "
            f"not '{args.backend}'"
        )

    if args.tol_record is not None and (
        args.backend != "cusparse"
        or args.application not in ("pca", "pca_lobpcg")
    ):
        # Only pca/grapp_cusparse.py and pca_lobpcg/grapp_cusparse_lobpcg.py define
        # --tol-record; forwarding it to any other script is an argparse error there.
        sys.exit(
            "error: --tol-record is only used by the cusparse backend for the "
            "pca and pca_lobpcg applications"
        )

    if args.bolt_pheno_file is not None and args.application != "bolt":
        sys.exit(
            f"error: --bolt-pheno-file is only used by the bolt application, "
            f"not '{args.application}'"
        )

    if args.output is not None and args.application == "kernel":
        sys.exit("error: --output is not supported by the kernel application")

    if args.output is not None and args.output.is_dir():
        # Every backend writes --output as a single result TSV (grapp to_csv,
        # boltlmm copyfile, plink to_csv), never into a directory.
        sys.exit(f"error: --output must be a file path, not a directory: {args.output}")

    if args.work_dir is not None and args.backend not in ("boltlmm", "plink"):
        sys.exit(
            f"error: --work-dir is only used by the boltlmm and plink baselines, "
            f"not '{args.backend}'"
        )

    if args.skip_output and (args.backend == "boltlmm" or args.application == "kernel"):
        sys.exit(
            "error: --skip-output is not supported by the boltlmm backend (always "
            "writes its stats) or the kernel application (writes no TSV)"
        )

    script_path, adapter = resolve_backend(args.application, args.backend)

    start_line = _line_count(args.record) if args.record is not None else 0

    for i in range(args.warmup):
        cmd = build_cmd(args, script_path, adapter, record=False)
        run_iteration(f"warmup {i + 1}/{args.warmup}", cmd)

    for i in range(args.runs):
        cmd = build_cmd(args, script_path, adapter, record=True)
        run_iteration(f"run {i + 1}/{args.runs}", cmd)

    if args.record is not None:
        print_summary(args.record, start_line)
    else:
        print("\nNo --record given; skipping aggregate summary.")


if __name__ == "__main__":
    main()
