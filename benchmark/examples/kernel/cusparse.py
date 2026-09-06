"""Standalone cuSPARSE matmul kernel benchmark.

Loads one or more `.grg_spmv` files (or a directory searched for `*.grg_spmv`)
directly via pygrgl_spmv (no grapp), then times only the sparse matmul kernel: a
concurrent sweep that dispatches one cuSPARSE SpMV/SpMM per chromosome across a
ThreadPool and measures the wall time of the whole sweep. Loading is excluded.

A single run benchmarks one direction (`--direction up|down`) at one width
(`--k`). `miss`/`init` inputs are not used (plain X @ V / X^T @ V).

Modes:
  (default)            eager cuSPARSE, numpy host input/output.
  --capture            replay pre-captured CUDA graphs, numpy host I/O (copy mode).
  --capture --native   GPU-resident I/O: inputs/outputs are cupy arrays, no host
                       transfers per matmul.
"""

import argparse
import concurrent.futures
import json
import logging
import pathlib
import threading
import time
from contextlib import ExitStack

import numpy
import cupy

from pygrgl_spmv import (
    make_backend_cusparse,
    make_runconfig_kernel,
    load_grg_spmv_multi,
)

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("kernel_cusparse")


def _resolve_inputs(inputs):
    paths = [pathlib.Path(p) for p in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        files = sorted(paths[0].glob("*.grg_spmv"))
        if not files:
            raise SystemExit(f"No *.grg_spmv files found in {paths[0]}")
        return files
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Input is not a file: {p}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="Either a single directory (searched for *.grg_spmv) or an explicit list of .grg_spmv files.")
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
    ap.add_argument("--capture", action="store_true",
                    help="Enable CUDA graph capture (replay captured graphs).")
    ap.add_argument("--native", action="store_true",
                    help="GPU-native I/O (requires --capture): inputs/outputs are cupy "
                         "arrays on device, no host transfers per matmul.")
    ap.add_argument("--allow-residency", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep the sparse layout GPU-resident (default). Pass "
                         "--no-allow-residency to force streaming mode, which "
                         "requires --vram-budget-mb > 0.")
    ap.add_argument("--vram-budget-mb", type=int, default=0,
                    help="GPU memory cap in MiB, applied only in streaming mode "
                         "(--no-allow-residency), where it must be > 0. Default: 0 (no cap).")
    ap.add_argument("--device-map", type=pathlib.Path, default=None,
                    help='JSON file mapping {"<file_stem>": {"cuda_device": int}, ...}. '
                         "Default: all chromosomes on GPU0.")
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
    if args.native and not args.capture:
        ap.error("--native requires --capture")
    if not args.allow_residency and args.vram_budget_mb <= 0:
        ap.error("--no-allow-residency (streaming) requires --vram-budget-mb > 0")

    paths = _resolve_inputs(args.inputs)
    path_strs = [str(p) for p in paths]

    # Eager (non-capture) cuSPARSE shares one CusparseRuntime per device and
    # rejects concurrent calls, so it cannot drive a multi-chromosome sweep.
    # Capture mode serializes same-device calls internally (CapturedBoundGRG).
    if not args.capture and len(paths) > 1:
        ap.error("non-capture mode supports only a single GRG; pass --capture "
                 f"to benchmark {len(paths)} chromosomes concurrently")
    np_dtype = numpy.float32 if args.dtype == "float32" else numpy.float64

    LOGGER.info("Inputs (%d):", len(paths))
    for p in paths:
        LOGGER.info("  %s", p)

    if args.device_map is not None:
        with args.device_map.open() as f:
            device = json.load(f)
        LOGGER.info("Device map: %s", device)
    else:
        device = 0
        LOGGER.warning("Device map not found, all chromosomes are on GPU0")

    backend = make_backend_cusparse(
        device=device,
        allow_residency=args.allow_residency,
        vram_budget_mb=args.vram_budget_mb,
        capture=args.capture,
        native=args.native,
    )
    # Capture mode consumes capture_ops; eager mode uses only req.req. Either way
    # the kernel runconfig is the bare matmul case (no init/miss, by_individual=False).
    req = make_runconfig_kernel(args.direction, args.k)

    with ExitStack() as stack:
        t_load0 = time.perf_counter()
        ops = load_grg_spmv_multi(path_strs, backend, req, stack, dtype=np_dtype)
        t_load = time.perf_counter() - t_load0

        n_grgs = len(ops)
        n_individuals = ops[0].num_individuals
        total_mutations = sum(o.num_mutations for o in ops)
        LOGGER.info("Loaded %d GRG(s) in %.3f s: %d individuals, %d total mutations",
                    n_grgs, t_load, n_individuals, total_mutations)

        rng = numpy.random.default_rng(2026)
        inputs_per_op = []
        for o in ops:
            cols = o.num_samples if args.direction == "up" else o.num_mutations
            v = rng.standard_normal((args.k, cols), dtype=numpy.float64).astype(np_dtype)
            if args.native:
                # Native matmul requires a cupy array on the op's device.
                dev = getattr(o, "device", 0) or 0
                with cupy.cuda.Device(dev):
                    v = cupy.asarray(v)
            inputs_per_op.append(v)

        # In capture mode CapturedBoundGRG.matmul serializes same-device calls and
        # synchronizes before returning; in eager mode there is only one GRG (see
        # the --capture guard above). So a plain ThreadPool sweep is safe, and
        # f.result() returns fully-materialized results — no extra sync needed.
        def do_matmul(op, v):
            return op.matmul(v, args.direction)

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

            def gated(op, v):
                ready.release()
                gate.wait()
                return do_matmul(op, v)

            futs = [pool.submit(gated, o, inputs_per_op[i]) for i, o in enumerate(ops)]
            for _ in range(n_workers):
                ready.acquire()
            t0 = time.perf_counter()
            gate.set()
            for f in futs:
                f.result()
            return (time.perf_counter() - t0) * 1000.0

        LOGGER.info("Benchmarking direction=%s k=%d dtype=%s threads=%d capture=%s native=%s "
                    "(warmup=%d trials=%d)",
                    args.direction, args.k, args.dtype, n_workers,
                    args.capture, args.native, args.warmup, args.trials)

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
