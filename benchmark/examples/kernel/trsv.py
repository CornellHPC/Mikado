"""Standalone cuSPARSE SpTrSV kernel benchmark (wrapper around the C++ binary).

Drives the `sptrsv` binary (built from code/trsv/) on a
single `.csr` triangular matrix M = I - A produced offline by
`code/trsv/grg_to_csr.py`. The binary solves op(M) x = b with
cuSPARSE SpSV (k=1) or SpSM (k>1), timing only the per-iteration solve (plus the
fair same-GPU input/output D2D copies); this wrapper parses its machine-readable
`RESULT_JSON:` line and appends a unified run record matching the other kernel
backends (see examples/kernel/cusparse.py).

Unlike the cusparse backend (which consumes `.grg_spmv` artifacts and sweeps all
chromosomes concurrently), this is a single-matrix, single-GPU benchmark: the
input must resolve to exactly one `.csr` file.
"""

import argparse
import json
import logging
import pathlib
import shutil
import subprocess
import sys

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("kernel_trsv")

# float32/float64 (numpy-style, matching cusparse.py) -> the binary's C++ names.
_DTYPE_TO_BIN = {"float32": "float", "float64": "double"}


def _resolve_input(inputs):
    """Resolve the dataset to exactly one .csr file.

    A single directory is searched for *.csr and must contain exactly one; an
    explicit path must be an existing .csr file. Mirrors cusparse.py's
    _resolve_inputs shape but enforces the single-matrix contract.
    """
    paths = [pathlib.Path(p) for p in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        files = sorted(paths[0].glob("*.csr"))
        if not files:
            raise SystemExit(f"No *.csr files found in {paths[0]}")
        if len(files) > 1:
            raise SystemExit(
                f"{paths[0]} contains {len(files)} *.csr files; the trsv backend "
                "benchmarks a single matrix. Pass one .csr file explicitly."
            )
        return files[0]
    if len(paths) != 1:
        raise SystemExit("trsv backend takes exactly one .csr input")
    p = paths[0]
    if not p.is_file():
        raise SystemExit(f"Input is not a file: {p}")
    if p.suffix != ".csr":
        raise SystemExit(f"Input must be a .csr file, got: {p}")
    return p


def _resolve_binary(explicit):
    """Locate the sptrsv binary: --sptrsv-bin, then PATH, then repo-relative."""
    if explicit is not None:
        cand = pathlib.Path(explicit)
        if not (cand.is_file() and shutil.which(str(cand.resolve()))):
            raise SystemExit(f"--sptrsv-bin is not an executable file: {cand}")
        return str(cand.resolve())
    on_path = shutil.which("sptrsv")
    if on_path:
        return on_path
    # parents[3] of benchmark/examples/kernel/trsv.py is the repo root.
    repo_bin = pathlib.Path(__file__).resolve().parents[3] / "code/trsv/sptrsv"
    if repo_bin.is_file():
        return str(repo_bin)
    raise SystemExit(
        "could not find the 'sptrsv' binary: pass --sptrsv-bin PATH, put it on "
        "PATH, or build it (make -C code/trsv)."
    )


def _parse_result_json(stdout):
    """Extract and parse the single 'RESULT_JSON: {...}' line from binary stdout."""
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):].strip())
    raise SystemExit("sptrsv produced no RESULT_JSON line; stdout:\n" + stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="A single .csr file, or a directory containing exactly one *.csr.")
    ap.add_argument("--direction", choices=["up", "down"], default="up",
                    help="Solve op(M) x = b: 'up' (M, lower-tri) or 'down' (M^T, "
                         "transpose). Default: up.")
    ap.add_argument("--k", type=int, default=1,
                    help="Number of RHS columns. k=1 -> cusparseSpSV, k>1 -> SpSM. Default: 1.")
    ap.add_argument("--trials", type=int, default=10,
                    help="Number of timed solve iterations. Default: 10.")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Number of untimed warmup iterations. Default: 3.")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                    help="Compute dtype. Default: float64.")
    ap.add_argument("--sptrsv-bin", dest="sptrsv_bin", default=None,
                    help="Path to the sptrsv binary (default: PATH, then repo-relative).")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    if args.k < 1:
        ap.error(f"--k must be >= 1, got {args.k}")
    if args.trials < 1:
        ap.error(f"--trials must be >= 1, got {args.trials}")
    if args.warmup < 0:
        ap.error(f"--warmup must be >= 0, got {args.warmup}")

    csr_path = _resolve_input(args.inputs)
    binary = _resolve_binary(args.sptrsv_bin)
    c_dtype = _DTYPE_TO_BIN[args.dtype]

    cmd = [
        binary, str(csr_path),
        "--dtype", c_dtype,
        "--k", str(args.k),
        "--dir", args.direction,
        "--iters", str(args.trials),
        "--warmup", str(args.warmup),
    ]
    LOGGER.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        LOGGER.error("sptrsv failed (exit %d)\nstdout:\n%s\nstderr:\n%s",
                     proc.returncode, proc.stdout, proc.stderr)
        raise SystemExit(proc.returncode)

    result = _parse_result_json(proc.stdout)
    trials_ms = list(result["solve_trials_ms"])
    arr = numpy.asarray(trials_ms, dtype=numpy.float64)
    mean_ms, std_ms, min_ms = float(arr.mean()), float(arr.std()), float(arr.min())
    LOGGER.info("solve: mean_ms=%.4f std_ms=%.4f min_ms=%.4f analysis_ms=%.4f residual=%.3e",
                mean_ms, std_ms, min_ms, result["analysis_ms"], result["residual"])

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="kernel",
            params=vars(args),
            input_files=input_files_summary([csr_path]),
            metrics={
                "matmul_mean_ms": mean_ms,
                "matmul_std_ms": std_ms,
                "matmul_min_ms": min_ms,
                "matmul_trials_ms": trials_ms,
                "analysis_ms": float(result["analysis_ms"]),
                "buffer_bytes": int(result["buffer_bytes"]),
                "residual": float(result["residual"]),
            },
            output_summary={
                "n": int(result["n"]),
                "nnz": int(result["nnz"]),
                "routine": result["routine"],
                "direction": args.direction,
                "k": int(args.k),
                "dtype": args.dtype,
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
