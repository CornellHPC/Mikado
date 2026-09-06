"""Standalone grapp + MKL PCA benchmark, LOBPCG solver.

LOBPCG variant of ``grapp_mkl.py``. Loads one or more `.grg_spmv` files (or a
directory searched for `*.grg_spmv`), combines them into a single
multi-chromosome PCA via `grapp.linalg.PCs(solver="lobpcg", ...)`, prints
load/compute/e2e timings, and writes the PC-score dataframe to a TSV.

The MKL backend performs GPU-free (CPU) SpMV; the LOBPCG iteration itself runs on
the host via SciPy. See ``grapp_grgl_lobpcg.py`` for the LOBPCG error-control
notes (``tol`` is a residual-norm tolerance, identical across backends for an
explicit ``tol > 0``; ``maxiter`` defaults to 100 here because the solver default
of 20 under-converges PCA).
"""

import argparse
import json
import logging
import pathlib
import time
from contextlib import ExitStack

import numpy

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


LOGGER = logging.getLogger("grapp_mkl_lobpcg")


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Either a single directory (searched for *.grg_spmv) or an explicit list of .grg_spmv files.")
    ap.add_argument("-d", "--dimensions", "--pcs", dest="pcs", type=int, default=10,
                    help="Number of principal components to extract. Default: 10.")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="LOBPCG residual-norm tolerance passed to PCs. 0 uses the solver "
                         "default (sqrt(eps)*n for the SciPy host iteration). Default: 0.")
    ap.add_argument("--maxiter", type=int, default=100,
                    help="LOBPCG maximum iterations. The solver default (20) often "
                         "under-converges PCA; default here: 100.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the LOBPCG init matrix X (n_individuals x k) passed to PCs "
                         "as init_matrix. Default: 0.")
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
    req = make_runconfig_pca(maxk=args.pcs)

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

        # LOBPCG starts from an (n_individuals x k) subspace, not a single vector.
        init_matrix = numpy.random.default_rng(args.seed).standard_normal(
            (n_individuals, args.pcs)
        )
        LOGGER.info("Generated LOBPCG init matrix X (%d x %d) from seed %d",
                    n_individuals, args.pcs, args.seed)

        t_cmp0 = time.perf_counter()
        pcs_df, eig_vals = PCs(
            grgs,
            k=args.pcs,
            threads=args.threads,
            solver="lobpcg",
            tol=args.tol,
            maxiter=args.maxiter,
            init_matrix=init_matrix,
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

    # MKL runs the LOBPCG iteration on the host (SciPy), so the residual tolerance
    # in effect when --tol <= 0 is SciPy's sqrt(eps_float64) * n.
    effective_tol = args.tol
    if args.tol <= 0.0:
        effective_tol = numpy.sqrt(numpy.finfo(numpy.float64).eps) * n_individuals

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
                "solver": "lobpcg",
                "maxiter": int(args.maxiter),
                "effective_tol": effective_tol,
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
