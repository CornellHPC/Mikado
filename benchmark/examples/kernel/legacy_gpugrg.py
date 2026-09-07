"""Standalone legacy GPUGRG matmul kernel benchmark.

Loads one or more precomputed `.gpugrg` files (or a directory searched for
`*.gpugrg`) directly via pygrgl's legacy GPU backend (`pygrgl.load_gpu_grg`, no
grapp, no per-run conversion), then times only the sparse matmul kernel: a
concurrent sweep that dispatches one `GPUGRG.matmul` per chromosome across a
ThreadPool and measures the wall time of the whole sweep. Loading is excluded.

The `.gpugrg` artifacts are produced offline by
`code/graph-first/grg_to_gpugrg.py` (which runs `pygrgl.grg_to_gpu` once and
serializes it), so the conversion cost is paid once rather than on every
benchmark run -- mirroring how cusparse.py / trsv.py consume precomputed
`.grg_spmv` / `.csr` artifacts.

A single run benchmarks one direction (`--direction up|down`) at one width
(`--k`). `miss`/`init` inputs are not used (plain X @ V / X^T @ V). Single-GPU:
the legacy GPUGRG Python API takes no device argument, so all chromosomes share
the default CUDA device.
"""

import argparse
import concurrent.futures
import logging
import pathlib
import threading
import time

import numpy
import pygrgl

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("kernel_legacy_gpugrg")


def _resolve_inputs(inputs):
    paths = [pathlib.Path(p) for p in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        files = sorted(paths[0].glob("*.gpugrg"))
        if not files:
            raise SystemExit(f"No *.gpugrg files found in {paths[0]}")
        return files
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Input is not a file: {p}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="Either a single directory (searched for *.gpugrg) or an explicit list of .gpugrg files.")
    ap.add_argument("--direction", choices=["up", "down"], default="up",
                    help="Matmul direction: 'up' (X @ V, V has num_samples rows) or "
                         "'down' (X^T @ V, V has num_mutations rows). Default: up.")
    ap.add_argument("--k", type=int, default=1,
                    help="Number of probe vectors (matmul width). k=1 is SpMV, k>1 SpMM. Default: 1.")
    ap.add_argument("--trials", type=int, default=10,
                    help="Number of timed sweeps. Default: 10.")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Number of untimed warmup sweeps. Default: 3.")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                    help="Input matrix dtype. Default: float64.")
    ap.add_argument("--threads", type=int, default=32,
                    help="ThreadPool workers for the per-chromosome sweep (capped at the "
                         "number of input GRGs). Default: 32.")
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

    paths = _resolve_inputs(args.inputs)
    path_strs = [str(p) for p in paths]
    np_dtype = numpy.float32 if args.dtype == "float32" else numpy.float64

    LOGGER.info("Inputs (%d):", len(paths))
    for p in paths:
        LOGGER.info("  %s", p)

    # --- Load (excluded from timing). load_gpu_grg allocates GPU memory and
    # copies the precomputed GPUGRG over; no grg_to_gpu conversion here. ---
    t_load0 = time.perf_counter()
    grgs = [pygrgl.load_gpu_grg(p) for p in path_strs]
    t_load = time.perf_counter() - t_load0

    n_grgs = len(grgs)
    n_individuals = grgs[0].num_individuals
    total_mutations = sum(g.num_mutations for g in grgs)
    LOGGER.info("Loaded %d GPUGRG(s) in %.3f s: %d individuals, %d total mutations",
                n_grgs, t_load, n_individuals, total_mutations)

    direction = (pygrgl.TraversalDirection.UP if args.direction == "up"
                 else pygrgl.TraversalDirection.DOWN)

    rng = numpy.random.default_rng(2026)
    inputs_per_grg = []
    for g in grgs:
        cols = g.num_samples if args.direction == "up" else g.num_mutations
        inputs_per_grg.append(rng.standard_normal((args.k, cols), dtype=numpy.float64).astype(np_dtype))

    def do_matmul(grg, v):
        return grg.matmul(v, direction)

    n_workers = min(args.threads, n_grgs)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)

    def sweep():
        # Startup gate: every worker rendezvouses on `ready`/`gate` before any
        # matmul runs. Timing begins only once all n_workers threads are created
        # and waiting, so thread-creation/submission overhead is excluded and all
        # chromosomes start together. Fresh gate+semaphore per sweep avoid stale
        # permits when n_grgs > n_workers.
        gate = threading.Event()
        ready = threading.Semaphore(0)

        def gated(grg, v):
            ready.release()
            gate.wait()
            return do_matmul(grg, v)

        futs = [pool.submit(gated, g, inputs_per_grg[i]) for i, g in enumerate(grgs)]
        for _ in range(n_workers):
            ready.acquire()
        t0 = time.perf_counter()
        gate.set()
        for f in futs:
            f.result()
        return (time.perf_counter() - t0) * 1000.0

    LOGGER.info("Benchmarking direction=%s k=%d dtype=%s threads=%d (warmup=%d trials=%d)",
                args.direction, args.k, args.dtype, n_workers, args.warmup, args.trials)

    for _ in range(args.warmup):
        sweep()
    times_ms = [sweep() for _ in range(args.trials)]
    pool.shutdown()

    arr = numpy.asarray(times_ms, dtype=numpy.float64)
    mean_ms, std_ms, min_ms = float(arr.mean()), float(arr.std()), float(arr.min())
    LOGGER.info("matmul sweep: mean_ms=%.4f std_ms=%.4f min_ms=%.4f", mean_ms, std_ms, min_ms)

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="kernel",
            params=vars(args),
            input_files=input_files_summary(paths),
            metrics={
                "matmul_mean_ms": mean_ms,
                "matmul_std_ms": std_ms,
                "matmul_min_ms": min_ms,
                "matmul_trials_ms": times_ms,
            },
            output_summary={
                "num_grgs": int(n_grgs),
                "num_individuals": int(n_individuals),
                "total_mutations": int(total_mutations),
                "direction": args.direction,
                "k": int(args.k),
                "dtype": args.dtype,
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
