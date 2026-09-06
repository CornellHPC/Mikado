"""Standalone grapp + MKL basic GWAS benchmark.

Loads a single `.grg_spmv` file with the MKL SpMV backend, runs a basic
(fixed-effect, per-SNP) linear-regression GWAS via
`grapp.assoc.linear_assoc_no_covar` / `linear_assoc_covar`, prints
load/compute/e2e timings, and writes the association dataframe to a TSV.
"""

import argparse
import json
import logging
import pathlib
import time
from contextlib import ExitStack

import numpy
import pandas

from pygrgl_spmv import (
    make_backend_mkl,
    make_runconfig_gwas,
    load_grg_spmv_single,
)
from grapp.assoc import (
    read_pheno,
    read_plink_covariates,
    linear_assoc_no_covar,
    linear_assoc_covar,
)
from grapp.grg_calculator import GRGSpMVCalculator

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("grapp_mkl")


def _resolve_mkl_threads_arg(value):
    """Resolve --mkl-threads into the form make_backend_mkl expects.

    An int value is the number of MKL threads to use. Otherwise the value is
    treated as a path to a JSON file mapping
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
    ap.add_argument("input", help="A single .grg_spmv file.")
    ap.add_argument("-p", "--phenotypes", default=None,
                    help="Phenotype file. If omitted, random phenotype values are generated.")
    ap.add_argument("-c", "--covariates", default=None,
                    help="Covariates file: plink format (.txt) or pandas dataframe (.tsv).")
    ap.add_argument("-b", "--binomial", action="store_true",
                    help="Use the binomial approximation for SNP variance instead of sample variance.")
    ap.add_argument("-s", "--standardize", action="store_true",
                    help="Standardize the X and Y matrices prior to regression.")
    ap.add_argument("-r", "--regress-y", dest="regress_y", action="store_true",
                    help="Regress covariates out of Y only (default: QR adjusts both X and Y).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the randomly generated phenotype (used only when --phenotypes is omitted). Default: 0.")
    ap.add_argument("--mkl-threads", dest="mkl_threads", type=str, default="1",
                    help="MKL SpMV threads. Either an int (default 1) or a path to a JSON file "
                         'mapping {"<file_stem>": {"mkl_threads": [n_up, n_down]}, ...}.')
    ap.add_argument("--optimize", action="store_true",
                    help="Enable MKL inspector-executor optimization in the MKL backend.")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("gwas_results.tsv"),
                    help="TSV path for the association dataframe (variants x stats).")
    ap.add_argument("--skip-output", dest="skip_output", action="store_true",
                    help="Skip writing the association TSV (timings still run).")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    path = pathlib.Path(args.input)
    if not path.is_file():
        raise SystemExit(f"Input is not a file: {path}")
    if args.regress_y and args.covariates is None:
        raise SystemExit("--regress-y only applies when --covariates is given")

    LOGGER.info("Input: %s", path)

    # Covariates (optional). Loaded up front because the runconfig's max UP width
    # (maxk) must cover the X^T Q product, whose k = n_covariates + 1 (intercept).
    if args.covariates is not None:
        if args.covariates.endswith(".txt"):
            C = read_plink_covariates(args.covariates)
        elif args.covariates.endswith(".tsv"):
            C = pandas.read_csv(args.covariates, delimiter="\t").to_numpy()
        else:
            raise SystemExit("Covariates filename must end in .txt (plink) or .tsv (pandas dataframe)")
        method = "regress" if args.regress_y else "QR"
        n_covariates = int(C.shape[1])
    else:
        C = None
        method = None
        n_covariates = 0

    dist = "binomial" if args.binomial else "sample"
    maxk = n_covariates + 1 if C is not None else 1

    n_threads = _resolve_mkl_threads_arg(args.mkl_threads)
    if isinstance(n_threads, int):
        LOGGER.info("MKL threads: %d", n_threads)
    else:
        LOGGER.info("MKL thread map: %s", n_threads)
    LOGGER.info("dist=%s standardize=%s covariates=%d method=%s maxk=%d",
                dist, args.standardize, n_covariates, method, maxk)

    backend = make_backend_mkl(n_threads=n_threads, optimize=args.optimize)
    req = make_runconfig_gwas(maxk=maxk, sample_variance=not args.binomial)

    t_e2e0 = time.perf_counter()
    with ExitStack() as stack:
        t_load0 = time.perf_counter()
        grg = GRGSpMVCalculator(
            load_grg_spmv_single(str(path), backend, req, stack)
        )
        t_load = time.perf_counter() - t_load0

        n_individuals = grg.num_individuals
        total_mutations = grg.num_mutations
        LOGGER.info("Loaded GRG: %d individuals, %d mutations", n_individuals, total_mutations)

        # Phenotype: read from file, or generate a reproducible random vector.
        if args.phenotypes is None:
            LOGGER.info("No phenotype provided; generating random phenotype (seed=%d)", args.seed)
            y = numpy.random.default_rng(args.seed).standard_normal(n_individuals)
        else:
            y = read_pheno(args.phenotypes)
            if len(y) != n_individuals:
                raise SystemExit(f"Phenotype file has {len(y)} rows, expected {n_individuals}")

        if C is not None and C.shape[0] != n_individuals:
            raise SystemExit(f"Covariate file has {C.shape[0]} rows, expected {n_individuals}")

        t_cmp0 = time.perf_counter()
        if C is not None:
            gwas_df = linear_assoc_covar(
                grg, y, C, method=method, standardize=args.standardize, dist=dist, return_raw=True
            )
        else:
            gwas_df = linear_assoc_no_covar(
                grg, y, standardize=args.standardize, dist=dist, return_raw=True
            )
        t_compute = time.perf_counter() - t_cmp0
    t_e2e = time.perf_counter() - t_e2e0

    LOGGER.info("Compute completed.")
    LOGGER.info("load:    %8.3f s", t_load)
    LOGGER.info("compute: %8.3f s", t_compute)
    LOGGER.info("e2e:     %8.3f s", t_e2e)

    n_results = int(len(gwas_df))
    if args.skip_output:
        LOGGER.info("Skipping output TSV (--skip-output)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gwas_df.to_csv(args.output, sep="\t", index=False)
        LOGGER.info("Wrote %s (%d variant results x %d columns)",
                    args.output, gwas_df.shape[0], gwas_df.shape[1])

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="gwas",
            params=vars(args),
            input_files=input_files_summary([path]),
            metrics={"load": t_load, "compute": t_compute, "e2e": t_e2e},
            output_summary={
                "num_individuals": int(n_individuals),
                "total_mutations": int(total_mutations),
                "num_covariates": n_covariates,
                "dist": dist,
                "standardize": bool(args.standardize),
                "maxk": int(maxk),
                "num_results": n_results,
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
