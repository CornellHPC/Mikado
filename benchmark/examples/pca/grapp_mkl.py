"""Standalone grapp + MKL PCA benchmark.

Loads one or more `.grg_spmv` files (or a directory searched for `*.grg_spmv`),
combines them into a single multi-chromosome PCA via `grapp.linalg.PCs`, prints
load/compute/e2e timings, and writes the PC-score dataframe to a TSV.
"""

import argparse
import json
import logging
import pathlib
import time
from contextlib import ExitStack

from pygrgl_spmv import (
    make_backend_mkl,
    make_runconfig_pca,
    load_grg_spmv_multi,
)
from grapp.grg_calculator import GRGSpMVCalculator
from grapp.linalg import PCs

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("grapp_mkl")


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


def _resolve_mkl_threads_arg(value):
    """Resolve --mkl-threads into the form make_backend_mkl expects.

    An int value is the number of MKL threads to use for every GRG. Otherwise
    the value is treated as a path to a JSON file mapping
    {"<file_stem>": {"mkl_threads": [n_up, n_down]}, ...}.
    """
    try:
        return int(value)
    except ValueError:
        p = pathlib.Path(value)
        if not p.is_file():
            raise SystemExit(f"--mkl-threads is neither an int nor a file: {value}")
        with p.open() as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="Either a single directory (searched for *.grg_spmv) or an explicit list of .grg_spmv files.")
    ap.add_argument("-d", "--dimensions", "--pcs", dest="pcs", type=int, default=10,
                    help="Number of principal components to extract. Default: 10.")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="Eigensolver convergence tolerance passed to PCs (0 = machine precision, rel error). Default: 0.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the eigensolver init vector (v0) generated and passed to PCs. Default: 0.")
    ap.add_argument("--mkl-threads", dest="mkl_threads", type=str, default="1",
                    help="MKL SpMV threads per GRG. Either an int (same count for every GRG; "
                         "default 1) or a path to a JSON file mapping "
                         '{"<file_stem>": {"mkl_threads": [n_up, n_down]}, ...}.')
    ap.add_argument("--optimize", action="store_true",
                    help="Enable MKL inspector-executor optimization in the MKL backend.")
    ap.add_argument("--threads", type=int, default=32,
                    help="Threads for PCs outer parallelism across GRGs (capped at the "
                         "number of input GRGs). Default: 32.")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("pca_results.tsv"),
                    help="TSV path for the PC-score dataframe (individuals x PCs).")
    ap.add_argument("--skip-output", dest="skip_output", action="store_true",
                    help="Skip writing the PC-score TSV (timings still run).")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    paths = _resolve_inputs(args.inputs)
    path_strs = [str(p) for p in paths]

    LOGGER.info("Inputs (%d):", len(paths))
    for p in paths:
        LOGGER.info("  %s", p)

    n_threads = _resolve_mkl_threads_arg(args.mkl_threads)
    if isinstance(n_threads, int):
        LOGGER.info("MKL threads: %d per GRG", n_threads)
    else:
        LOGGER.info("MKL thread map: %s", n_threads)

    backend = make_backend_mkl(n_threads=n_threads, optimize=args.optimize)
    req = make_runconfig_pca()

    t_e2e0 = time.perf_counter()
    with ExitStack() as stack:
        t_load0 = time.perf_counter()
        grgs = [
            GRGSpMVCalculator(g)
            for g in load_grg_spmv_multi(path_strs, backend, req, stack)
        ]
        t_load = time.perf_counter() - t_load0

        n_grgs = len(grgs)
        n_individuals = grgs[0].num_individuals
        total_mutations = sum(g.num_mutations for g in grgs)
        LOGGER.info("Loaded %d GRG(s): %d individuals, %d total mutations",
                    n_grgs, n_individuals, total_mutations)

        import numpy
        init_vector = numpy.random.default_rng(args.seed).standard_normal(n_individuals)
        LOGGER.info("Generated eigsh init vector v0 (len=%d) from seed %d",
                    n_individuals, args.seed)

        t_cmp0 = time.perf_counter()
        pcs_df, eig_vals = PCs(
            grgs,
            k=args.pcs,
            threads=args.threads,
            tol=args.tol,
            init_vector=init_vector,
        )
        t_compute = time.perf_counter() - t_cmp0
    t_e2e = time.perf_counter() - t_e2e0

    LOGGER.info("Compute completed.")
    LOGGER.info("load:    %8.3f s", t_load)
    LOGGER.info("compute: %8.3f s", t_compute)
    LOGGER.info("e2e:     %8.3f s", t_e2e)

    if args.skip_output:
        LOGGER.info("Skipping output TSV (--skip-output)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pcs_df.to_csv(args.output, sep="\t")
        LOGGER.info("Wrote %s (%d individuals x %d PCs)",
                    args.output, pcs_df.shape[0], pcs_df.shape[1])

    effective_tol = args.tol
    if args.tol == 0.0:
        import numpy
        effective_tol = numpy.finfo(numpy.float64).eps * float(eig_vals[0])

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="pca",
            params=vars(args),
            input_files=input_files_summary(paths),
            metrics={"load": t_load, "compute": t_compute, "e2e": t_e2e},
            output_summary={
                "num_individuals": int(n_individuals),
                "num_pcs": int(args.pcs),
                "num_grgs": int(n_grgs),
                "total_mutations": int(total_mutations),
                "eig_vals": eig_vals,
                "effective_tol": effective_tol
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
