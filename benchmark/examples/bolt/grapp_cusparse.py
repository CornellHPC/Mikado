"""Standalone grapp + cuSPARSE BOLT-LMM-inf benchmark.

Loads one or more `.grg_spmv` files, runs `bolt_lmm_inf` against a synthetic
intercept-only model, prints timings and summary statistics, and writes the
per-variant result dataframe to a TSV.
"""

import argparse
import json
import logging
import math
import pathlib
import re
import time
from contextlib import ExitStack
import cupy
import torch

import numpy as np

from pygrgl_spmv import (
    make_backend_cusparse,
    make_runconfig_bolt,
    load_grg_spmv_multi,
)
from grapp.grg_calculator import GRGSpMVCalculator
from grapp.assoc.bolt_lmm import bolt_lmm_inf, lmm_inf_stats_to_dataframe
from grapp.assoc.bolt_inf_core import CovariateBasis
from grapp.assoc import read_pheno, read_plink_covariates

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


_CHR_RE = re.compile(r"chr(\d+)", re.IGNORECASE)

LOGGER = logging.getLogger("grapp_cusparse")


def _bolt_stream_float(value: float) -> str:
    return format(float(value), ".6g")


def _bolt_pvalue(value: float, stat: float) -> str:
    p_value = float(value)
    if p_value != 0.0:
        return f"{p_value:.1E}"
    log10p = math.log10(2.0) - math.log10(math.e) * float(stat) / 2.0 - 0.5 * math.log10(float(stat) * 2.0 * math.pi)
    exponent = math.floor(log10p)
    fraction = math.pow(10.0, log10p - exponent)
    if fraction >= 9.95:
        fraction = 1.0
        exponent += 1
    return f"{fraction:.1f}E{exponent:d}"


def _chrom_label(path: pathlib.Path):
    m = _CHR_RE.search(path.stem)
    return int(m.group(1)) if m else path.stem


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
    ap.add_argument("--input-seed", type=int, default=2026,
                    help="Phenotype RNG seed (used only when --pheno-file is not given).")
    ap.add_argument("--pheno-file", type=pathlib.Path, default=None,
                    help="External phenotype file (PLINK-style 'FID IID <col>...'; the last "
                         "column is the phenotype, read in file order to match GRG individual "
                         "order). If given, used instead of the synthetic seed-generated phenotype.")
    ap.add_argument("--bolt-seed", type=int, default=2026,
                    help="Seed for bolt_lmm_inf variance-component and calibration computation (default: 2026).")
    ap.add_argument("-c", "--covariates", type=pathlib.Path, default=None,
                    help="Covariate file (PLINK .txt format; FID/IID then covariate "
                         "columns; read in file order to match GRG individual order). "
                         "If omitted, an intercept-only model is used.")
    ap.add_argument("--covar-cols", nargs="*", default=(),
                    help="Names of categorical covariate columns.")
    ap.add_argument("--q-covar-cols", nargs="*", default=(),
                    help="Names of quantitative covariate columns.")
    ap.add_argument("--covar-max-levels", type=int, default=10,
                    help="Maximum number of levels for categorical covariates (default: 10).")
    ap.add_argument("--capture", action="store_true",
                    help="Enable CUDA graph capture in cuSPARSE backend.")
    ap.add_argument("--native", action="store_true",
                    help="Enable GPU-native I/O (requires --capture). NOTE: for "
                         "exact run-to-run reproducibility on MULTI-DEVICE native "
                         "runs of near-zero-heritability phenotypes, set the env "
                         "var CUDA_LAUNCH_BLOCKING=1. Without it the result stays "
                         "correct in expectation but the degenerate ~0-h2 REML fit "
                         "can be perturbed by unavoidable cross-GPU FP atomic "
                         "non-determinism.")
    ap.add_argument("--force-spmm", dest="force_spmm",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Capture graphs at k=2 (SpMM path) instead of k=1 (SpMV). "
                         "Default: off (--force-spmm to enable).")
    ap.add_argument("--device-map", type=pathlib.Path, default=None,
                    help='JSON file mapping {"<file_stem>": {"cuda_device": int}, ...}.')
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("grapp_cusparse_results.tsv"),
                    help="TSV path for the per-variant results dataframe.")
    ap.add_argument("--rewrite-output", dest="rewrite_output",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Reformat the output columns into BOLT-LMM stream style "
                         "(_bolt_stream_float / _bolt_pvalue). Use --no-rewrite-output "
                         "to write the raw dataframe values instead (default: no rewrite).")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    ap.add_argument("--skip-output", dest="skip_output", action="store_true",
                    help="Skip writing the per-variant results TSV (timings/record still run).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    if args.native and not args.capture:
        ap.error("--native requires --capture")

    paths = _resolve_inputs(args.inputs)
    chrom_labels = [_chrom_label(p) for p in paths]
    path_strs = [str(p) for p in paths]

    LOGGER.info("Inputs (%d):", len(paths))
    for c, p in zip(chrom_labels, paths):
        LOGGER.info("  chrom=%s  %s", c, p)

    if args.device_map is not None:
        with args.device_map.open() as f:
            device = json.load(f)
        LOGGER.info("Device map: %s", device)
    else:
        device = 0
        LOGGER.warning("Device map not found, all chromosomes are on GPU0")

    backend = make_backend_cusparse(device=device, capture=args.capture, native=args.native)
    req = make_runconfig_bolt(force_spmm=args.force_spmm)

    t_e2e0 = time.perf_counter()
    with ExitStack() as stack:
        t_load0 = time.perf_counter()
        grgs = [
            GRGSpMVCalculator(g)
            for g in load_grg_spmv_multi(path_strs, backend, req, stack)
        ]
        t_load = time.perf_counter() - t_load0

        n = grgs[0].num_individuals
        if args.covariates is not None:
            covar_mat = read_plink_covariates(str(args.covariates))
            if covar_mat.shape[0] != n:
                raise SystemExit(
                    f"Covariate file has {covar_mat.shape[0]} rows != num_individuals "
                    f"{n} (does {args.covariates} match the GRG individuals?)")
            covariates = CovariateBasis.from_matrix(
                np.column_stack([np.ones(n), covar_mat]),
                covar_cols=args.covar_cols,
                q_covar_cols=args.q_covar_cols,
                covar_max_levels=args.covar_max_levels,
            )
            LOGGER.info("Using covariates %s (N=%d, K=%d)",
                        args.covariates, covar_mat.shape[0], covar_mat.shape[1])
        else:
            covariates = CovariateBasis.intercept_only(n)
        if args.pheno_file is not None:
            y = read_pheno(str(args.pheno_file))
            if y.shape[0] != n:
                raise SystemExit(
                    f"phenotype length {y.shape[0]} != num_individuals {n} "
                    f"(does {args.pheno_file} match the GRG individuals?)")
            LOGGER.info("Using external phenotype %s (N=%d)", args.pheno_file, y.shape[0])
        else:
            y = np.random.default_rng(args.input_seed).standard_normal(n)
        chrom_grgs = list(zip(chrom_labels, grgs))

        t_cmp0 = time.perf_counter()
        fit, cal, _, stats = bolt_lmm_inf(
            chrom_grgs, y, covariates,
            seed=args.bolt_seed, threads=args.threads,
        )
        t_compute = time.perf_counter() - t_cmp0
    t_e2e = time.perf_counter() - t_e2e0

    LOGGER.info("Compute Completed; Converting output.")

    LOGGER.info("load:    %8.3f s", t_load)
    LOGGER.info("compute: %8.3f s", t_compute)
    LOGGER.info("e2e:     %8.3f s", t_e2e)
    LOGGER.info("h2=%.4f  sigma_g2=%.4f  calibration=%.4f",
                fit.h2, fit.sigma_g2, cal.factor)

    if args.skip_output:
        LOGGER.info("Skipping output TSV (--skip-output)")
    else:
        df = lmm_inf_stats_to_dataframe(stats, chrom_grgs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.rewrite_output:
            out_df = df.copy()
            for col in ("A1FREQ", "CHISQ_LINREG", "BETA", "SE", "CHISQ_BOLT_LMM_INF"):
                out_df[col] = out_df[col].map(_bolt_stream_float)
            out_df["P_LINREG"] = [
                _bolt_pvalue(p, s) for p, s in zip(df["P_LINREG"], df["CHISQ_LINREG"])
            ]
            out_df["P_BOLT_LMM_INF"] = [
                _bolt_pvalue(p, s) for p, s in zip(df["P_BOLT_LMM_INF"], df["CHISQ_BOLT_LMM_INF"])
            ]
        else:
            out_df = df
        out_df.to_csv(args.output, sep="\t", index=False)
        LOGGER.info("Wrote %s", args.output)

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="bolt",
            params=vars(args),
            input_files=input_files_summary(paths),
            metrics={"load": t_load, "compute": t_compute, "e2e": t_e2e},
            output_summary={
                "h2": fit.h2,
                "sigma_g2": fit.sigma_g2,
                "calibration": cal.factor
            },
        )


if __name__ == "__main__":
    main()
