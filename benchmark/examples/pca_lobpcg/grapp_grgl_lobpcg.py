"""Standalone grapp + pygrgl PCA benchmark, LOBPCG solver.

LOBPCG variant of ``grapp_grgl.py``. Loads one or more `.grg` files (or a
directory searched for `*.grg`), combines them into a single multi-chromosome
PCA via `grapp.linalg.PCs(solver="lobpcg", ...)`, prints load/compute/e2e
timings, and writes the PC-score dataframe to a TSV.

Error control (LOBPCG vs eigsh)
-------------------------------
Unlike eigsh, whose ``tol`` is an eigenvalue-accuracy criterion (relative for the
SciPy/CPU backend, absolute for the CuPy/GPU backend), LOBPCG's ``tol`` is a
**residual-norm** tolerance: an eigenpair (x, lambda) is converged once
``||A x - lambda x|| <= tol``. The SciPy and CuPy LOBPCG implementations apply
this criterion identically, so passing an explicit ``--tol > 0`` gives an
apples-to-apples error target across all three backends (grgl / mkl / cusparse).

The only cross-backend divergence is the *default* tolerance used when
``tol <= 0``: SciPy uses ``sqrt(eps_float64) * n ~= 1.49e-8 * n`` while CuPy
hardcodes ``sqrt(1e-15) * n ~= 3.16e-8 * n`` (~2.1x looser). Here ``n`` is the
operator dimension = number of individuals (the default ``XX^T`` PCA path). For
reproducible cross-backend comparisons, prefer an explicit ``--tol``.

LOBPCG's ``maxiter`` default (20) frequently under-converges PCA problems, so
this script defaults ``--maxiter`` to 100 and exposes it as a knob.
"""

import argparse
import logging
import pathlib
import time

import numpy

from grapp.grg_calculator import load_grg_calculator
from grapp.linalg import PCs

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("grapp_grgl_lobpcg")


def _resolve_inputs(inputs):
    paths = [pathlib.Path(p) for p in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        files = sorted(paths[0].glob("*.grg"))
        if not files:
            raise SystemExit(f"No *.grg files found in {paths[0]}")
        return files
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Input is not a file: {p}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Either a single directory (searched for *.grg) or an explicit list of .grg files.")
    ap.add_argument("-d", "--dimensions", "--pcs", dest="pcs", type=int, default=10,
                    help="Number of principal components to extract. Default: 10.")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="LOBPCG residual-norm tolerance passed to PCs. 0 uses the solver "
                         "default (sqrt(eps)*n for SciPy). Default: 0.")
    ap.add_argument("--maxiter", type=int, default=100,
                    help="LOBPCG maximum iterations. The solver default (20) often "
                         "under-converges PCA; default here: 100.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the LOBPCG init matrix X (n_individuals x k) passed to PCs "
                         "as init_matrix. Default: 0.")
    ap.add_argument("--threads", type=int, default=32)
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

    t_e2e0 = time.perf_counter()
    t_load0 = time.perf_counter()
    grgs = [load_grg_calculator(p) for p in path_strs]
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

    # LOBPCG residual tolerance actually in effect (SciPy/CPU backend here).
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
