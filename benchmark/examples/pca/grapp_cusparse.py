"""Standalone grapp + cuSPARSE PCA benchmark.

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
import cupy
import torch

from pygrgl_spmv import (
    make_backend_cusparse,
    make_runconfig_pca,
    load_grg_spmv_multi,
)
from grapp.grg_calculator import GRGSpMVCalculator
from grapp.linalg import PCs

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary


LOGGER = logging.getLogger("grapp_cusparse")


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


def _lookup_tol_from_record(record_path, paths, pcs, seed):
    """Find effective_tol in record_path for the case matching (paths, pcs, seed).

    Matches on the set of input-file (name, size) pairs plus pcs and seed.
    Builds the current run's (name, size) set via input_files_summary, the same
    helper that wrote the records, so the comparison is apples-to-apples.
    Raises SystemExit if no record matches, or if matching records disagree
    on effective_tol.
    """
    want_files = frozenset(
        (d["name"], d["size"]) for d in input_files_summary(paths)
    )
    matches = []  # (timestamp, effective_tol, lineno)
    with open(record_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = rec.get("params") or {}
            if params.get("pcs") != pcs or params.get("seed") != seed:
                continue
            rec_files = frozenset(
                (d.get("name"), d.get("size"))
                for d in (rec.get("input_files") or [])
            )
            if rec_files != want_files:
                continue
            tol = (rec.get("output_summary") or {}).get("effective_tol")
            if tol is None:
                continue
            matches.append((rec.get("timestamp"), float(tol), lineno))

    if not matches:
        raise SystemExit(
            f"--tol-record: no record in {record_path} matches "
            f"inputs={sorted(want_files)} pcs={pcs} seed={seed}"
        )

    tols = [t for _, t, _ in matches]
    lo, hi = min(tols), max(tols)
    if hi > 0 and (hi - lo) / hi > 1e-6:
        raise SystemExit(
            f"--tol-record: {len(matches)} matching records in {record_path} "
            f"disagree on effective_tol (min={lo!r}, max={hi!r}); refusing to guess"
        )
    # Use the most recent matching record's value.
    chosen = max(matches, key=lambda m: (m[0] or "", m[2]))[1]
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="Either a single directory (searched for *.grg_spmv) or an explicit list of .grg_spmv files.")
    ap.add_argument("-d", "--dimensions", "--pcs", dest="pcs", type=int, default=10,
                    help="Number of principal components to extract. Default: 10.")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="Eigensolver convergence tolerance passed to PCs (0 = machine precision, rel error for numpy and abs error for cupy). Default: 0.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the eigensolver init vector (v0) generated and passed to PCs. Default: 0.")
    ap.add_argument("--tol-record", dest="tol_record", type=pathlib.Path, default=None,
                    help="Path to a record JSONL (e.g. an MKL run record). Looks up the "
                         "effective_tol of the record whose inputs (name+size), pcs, and "
                         "seed match this run, and uses it as --tol. Native mode only; "
                         "mutually exclusive with a nonzero --tol.")
    ap.add_argument("--capture", action="store_true",
                    help="Enable CUDA graph capture in cuSPARSE backend.")
    ap.add_argument("--native", action="store_true",
                    help="Enable GPU-native I/O (requires --capture). Runs fully GPU-resident "
                         "eigsh via PCs(use_cupy=True); without it eigsh runs on the host with "
                         "GPU SpMV (host transfers per matmul).")
    ap.add_argument("--force-spmm", dest="force_spmm",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Capture graphs at k=2 (SpMM path) instead of k=1 (SpMV). "
                         "Default: off (--force-spmm to enable).")
    ap.add_argument("--device-map", type=pathlib.Path, default=None,
                    help='JSON file mapping {"<file_stem>": {"cuda_device": int}, ...}.')
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

    if args.native and not args.capture:
        ap.error("--native requires --capture")

    if args.tol_record is not None:
        if not args.native:
            ap.error("--tol-record requires --native")
        if args.tol != 0.0:
            ap.error("--tol-record cannot be combined with an explicit nonzero --tol")
        if not args.tol_record.is_file():
            ap.error(f"--tol-record file not found: {args.tol_record}")

    paths = _resolve_inputs(args.inputs)
    path_strs = [str(p) for p in paths]

    if args.tol_record is not None:
        args.tol = _lookup_tol_from_record(args.tol_record, paths, args.pcs, args.seed)
        LOGGER.info("Using effective_tol %.6e from record %s as --tol",
                    args.tol, args.tol_record)

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

    backend = make_backend_cusparse(device=device, capture=args.capture, native=args.native)
    req = make_runconfig_pca(force_spmm=args.force_spmm)

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
        if args.native:
            # Native runs do eigsh on the GPU (cupyx eigsh), so v0 must be a
            # device array; PCs passes init_vector through to eigsh unchanged.
            import cupy
            init_vector = cupy.asarray(init_vector)
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
        if args.native:
            effective_tol = numpy.finfo(numpy.float64).eps 
        else:
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
