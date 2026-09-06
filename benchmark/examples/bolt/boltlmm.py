"""Standalone original BOLT-LMM v2.5 --lmmInfOnly benchmark.

Self-contained: only requires stdlib + numpy + a pre-built BOLT-LMM v2.5
binary (path supplied via --bolt-bin). No imports from grg-spmv.

Drives the official BOLT-LMM binary against user-supplied PLINK BED/BIM/FAM
with a synthetic intercept-only phenotype.

BIM SNP IDs are rewritten to `chrom:bp:allele1:allele0` by default (matching
the e2e driver) so BOLT does not mask variants for duplicate IDs. Pass
`--no-rewrite-bim` to disable.


"""

import argparse
import collections
from email.policy import default
import logging
import os
import pathlib
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

import numpy as np

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary

LOGGER = logging.getLogger("boltlmm")


DEFAULT_NUM_CALIB_SNPS = 30
DEFAULT_H2_EST_MC_TRIALS = 0
DEFAULT_CG_TOL = 5e-4
DEFAULT_MAX_ITERS = 500
DEFAULT_BOLT_SEED = 2026


@dataclass(frozen=True)
class PlinkTriple:
    bim: pathlib.Path
    bed: pathlib.Path
    fam: pathlib.Path


def _triple_from_entry(entry) -> PlinkTriple:
    p = pathlib.Path(entry)
    bim = p if p.suffix == ".bim" else p.with_suffix(".bim")
    bed = bim.with_suffix(".bed")
    fam = bim.with_suffix(".fam")
    for f in (bim, bed, fam):
        if not f.exists():
            raise FileNotFoundError(f)
    return PlinkTriple(bim=bim, bed=bed, fam=fam)


def _glob_chrom_bim(plink_dir: pathlib.Path, chrom: str) -> pathlib.Path:
    token = f"chr{chrom}"
    patterns = (
        plink_dir / token / f"{token}.bim",
        plink_dir / f"{token}.bim",
        plink_dir / f"{token}.*.bim",
        plink_dir / f"{token}_*.bim",
        plink_dir / f"{token}-*.bim",
        plink_dir / f"*.{token}.*.bim",
        plink_dir / f"*.{token}_*.bim",
        plink_dir / f"*.{token}-*.bim",
    )
    matches = []
    for pat in patterns:
        matches.extend(sorted(pat.parent.glob(pat.name)))
    unique = tuple(dict.fromkeys(p.resolve() for p in matches if p.exists()))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one chr{chrom} BIM in {plink_dir}, found {len(unique)}"
        )
    return unique[0]


def resolve_plink(args) -> tuple[PlinkTriple, ...]:
    if args.plink is not None:
        return tuple(_triple_from_entry(e) for e in args.plink)
    chroms = tuple(c.strip() for c in args.chromosomes.split(",") if c.strip())
    return tuple(
        _triple_from_entry(_glob_chrom_bim(args.plink_dir, c)) for c in chroms
    )


def read_fam(path: pathlib.Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split()
            if len(fields) < 2:
                raise ValueError(f"invalid FAM line {lineno} in {path}: expected at least FID IID")
            samples.append((fields[0], fields[1]))
    if not samples:
        raise ValueError(f"FAM file is empty: {path}")
    return samples


def assert_same_fam(fam_paths) -> list[tuple[str, str]]:
    paths = [pathlib.Path(p) for p in fam_paths]
    if not paths:
        raise ValueError("at least one FAM path is required")
    first = read_fam(paths[0])
    for path in paths[1:]:
        if read_fam(path) != first:
            raise ValueError(f"FAM sample order differs between {paths[0]} and {path}")
    return first


def write_pheno(samples: list[tuple[str, str]], y: np.ndarray, path: pathlib.Path) -> None:
    if len(samples) != int(y.size):
        raise ValueError(f"phenotype length {y.size} does not match FAM sample count {len(samples)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("FID IID PHENO\n")
        for (fid, iid), value in zip(samples, y, strict=True):
            handle.write(f"{fid} {iid} {float(value):.17g}\n")


def _count_bim_rows(path: pathlib.Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def rewrite_bim(src: pathlib.Path, dst: pathlib.Path) -> tuple[pathlib.Path, list[int]]:
    """Rewrite SNP IDs to chrom:bp:allele1:allele0; skip duplicate-key rows with a warning.

    Returns (dst, skipped_row_indices). Skipped indices are 0-based BIM row numbers.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    key_counts: dict[str, int] = {}
    with src.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, start=1):
            fields = line.rstrip("\n").split()
            if len(fields) < 6:
                raise ValueError(f"{src}:{lineno} has fewer than 6 BIM fields")
            chrom, _id, _gpos, bp, a1, a0 = fields[:6]
            key = f"{chrom}:{bp}:{a1}:{a0}"
            key_counts[key] = key_counts.get(key, 0) + 1

    dup_keys = {k for k, c in key_counts.items() if c > 1}
    skipped: list[int] = []

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            fields = line.rstrip("\n").split()
            chrom, _id, gpos, bp, a1, a0 = fields[:6]
            new_id = f"{chrom}:{bp}:{a1}:{a0}"
            if new_id in dup_keys:
                skipped.append(idx)
                continue
            fout.write(f"{chrom}\t{new_id}\t{gpos}\t{bp}\t{a1}\t{a0}\n")

    if skipped:
        sample = ", ".join(
            f"{k} (x{key_counts[k]})" for k in list(dup_keys)[:5]
        )
        LOGGER.warning(
            "%s: skipped %d rows in %d duplicate-key groups (e.g., %s)",
            src.name, len(skipped), len(dup_keys), sample,
        )
    return dst, skipped


def write_filtered_bed(
    src: pathlib.Path, dst: pathlib.Path, *,
    n_individuals: int, skip_rows: set[int], total_rows: int,
) -> pathlib.Path:
    """Copy PLINK BED to dst, omitting variant rows whose 0-based index is in skip_rows."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpr = (n_individuals + 3) // 4
    with src.open("rb") as fin, dst.open("wb") as fout:
        magic = fin.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"unexpected BED magic in {src}: {magic!r}")
        fout.write(magic)
        for row in range(total_rows):
            chunk = fin.read(bpr)
            if len(chunk) != bpr:
                raise ValueError(f"short read at row {row} of {src}")
            if row in skip_rows:
                continue
            fout.write(chunk)
    return dst


def run_official_bolt(
    *,
    bolt_bin: pathlib.Path,
    fam: pathlib.Path,
    bims,
    beds,
    pheno: pathlib.Path,
    pheno_col: str,
    stats_file: pathlib.Path,
    num_threads: int,
    total_snps: int,
    num_leave_out_chunks: int,
    bolt_seed: int,
) -> None:
    cmd = [
        str(bolt_bin),
        "--fam", str(fam),
        "--phenoFile", str(pheno),
        "--phenoCol", pheno_col,
        "--lmmInfOnly",
        "--verboseStats",
        "--statsFile", str(stats_file),
        "--numThreads", str(int(num_threads)),
        # "--numLeaveOutChunks", str(int(num_leave_out_chunks)), should be unnecessary for more than 1 chunks
        # "--numCalibSnps", str(min(DEFAULT_NUM_CALIB_SNPS, max(1, total_snps))), default at 30
        # "--h2EstMCtrials", str(DEFAULT_H2_EST_MC_TRIALS), default to 0
        # "--CGtol", str(DEFAULT_CG_TOL),
        # "--maxIters", str(DEFAULT_MAX_ITERS), default is 500
        "--seed", str(int(bolt_seed)),
        "--noMapCheck",
        "--maxModelSnps", str(max(1, total_snps) + 10),
    ]
    for bim in bims:
        cmd.extend(["--bim", str(bim)])
    for bed in beds:
        cmd.extend(["--bed", str(bed)])

    LOGGER.info("Running official BOLT: %s", " ".join(cmd))
    log_path = pathlib.Path(stats_file).with_suffix(".log")
    # Stream BOLT's output line-by-line: tee each line to the console and the log
    # file (flushed) as it arrives, so progress is visible in realtime instead of
    # buffered until the run finishes. Keep a tail for the error message.
    tail: collections.deque[str] = collections.deque(maxlen=80)
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc, log_path.open("w", encoding="utf-8") as log_handle:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
            tail.append(line)
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"official BOLT failed with exit {returncode}; log: {log_path}\n"
            f"{''.join(tail)}"
        )


_TIME_FOR_RE = re.compile(r"^Time for (.+?)\s*=\s*([0-9.eE+-]+)\s*sec", re.MULTILINE)
_TOTAL_ELAPSED_RE = re.compile(r"^Total elapsed time for analysis\s*=\s*([0-9.eE+-]+)\s*sec", re.MULTILINE)
_H2_RE = re.compile(r"Estimated \(pseudo-\)heritability:\s*h2g\s*=\s*([0-9.eE+-]+)")
_SIGMA_RE = re.compile(r"Variance params:\s*sigma\^2_K\s*=\s*([0-9.eE+-]+)")
_CALIB_RE = re.compile(r"Calibration:\s*([0-9.eE+-]+)")

# Key under which BOLT's output-writing stage is recorded; excluded from e2e_no_write.
WRITE_OUTPUT_KEY = "streaming_genotypes_and_writing_output"


def _normalize_stage(name: str) -> str:
    """Turn a BOLT stage label into a snake_case dict key."""
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-z]+", "_", name)
    return name.strip("_")


def parse_bolt_log(log_path: pathlib.Path) -> tuple[dict[str, float], dict]:
    """Scrape staged timings and summary stats from a BOLT log.

    Returns ``(bolt_staged, bolt_summary)``. ``bolt_staged`` maps normalized
    snake_case stage names to seconds and includes ``total_elapsed``. ``bolt_summary``
    holds ``h2``/``sigma_g2``/``calibration`` (each ``None`` if not found). Missing
    lines are simply omitted; this function does not raise on a partial log.
    """
    text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")

    staged: dict[str, float] = {}
    for label, value in _TIME_FOR_RE.findall(text):
        staged[_normalize_stage(label)] = float(value)
    m = _TOTAL_ELAPSED_RE.search(text)
    if m:
        staged["total_elapsed"] = float(m.group(1))

    def _first(rx):
        mm = rx.search(text)
        return float(mm.group(1)) if mm else None

    summary = {
        "h2": _first(_H2_RE),
        "sigma2_K": _first(_SIGMA_RE),
        "calibration": _first(_CALIB_RE),
    }
    return staged, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plink", nargs="+",
                      help="List mode: PLINK .bim paths (or stems). Sibling .bed/.fam must exist.")
    mode.add_argument("--plink-dir", type=pathlib.Path,
                      help="Folder mode: directory containing chr<N>.{bim,bed,fam}.")
    ap.add_argument("--chromosomes",
                    help="Comma-separated chromosomes (required with --plink-dir).")
    ap.add_argument("--bolt-bin", type=pathlib.Path, required=True,
                    help="Path to the BOLT-LMM v2.5 binary (e.g., <cache>/BOLT-LMM_v2.5/src/bolt).")
    ap.add_argument("--work-dir", type=pathlib.Path, required=True,
                    help="Working directory for intermediate files.")
    ap.add_argument("--input-seed", type=int, default=2026,
                    help="Phenotype RNG seed (used only when --pheno-file is not given).")
    ap.add_argument("--pheno-file", type=pathlib.Path, default=None,
                    help="External phenotype file (PLINK-style 'FID IID <col>...'). If "
                         "given, it is passed to BOLT directly instead of the synthetic "
                         "seed-generated phenotype.")
    ap.add_argument("--pheno-col", default="PHENO",
                    help="Phenotype column name within --pheno-file (default: PHENO).")
    ap.add_argument("--bolt-seed", type=int, default=DEFAULT_BOLT_SEED,
                    help=f"BOLT --seed (default {DEFAULT_BOLT_SEED}, matches core.py).")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-rewrite-bim", action="store_true",
                    help="Skip the chrom:bp:a1:a0 SNP-ID rewrite (default: rewrite ON).")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("bolt_lmm_official_results.tsv"))
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    if args.plink_dir is not None and not args.chromosomes:
        ap.error("--chromosomes is required with --plink-dir")

    bolt_bin = args.bolt_bin.expanduser().resolve()
    if not bolt_bin.is_file():
        ap.error(f"--bolt-bin not found: {bolt_bin}")
    if not os.access(bolt_bin, os.X_OK):
        ap.error(f"--bolt-bin is not executable: {bolt_bin}")

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    t_e2e0 = time.perf_counter()

    t_prep0 = time.perf_counter()

    plink_files = resolve_plink(args)
    LOGGER.info("PLINK inputs (%d):", len(plink_files))
    for p in plink_files:
        LOGGER.info("  %s", p.bim)

    samples = assert_same_fam(p.fam for p in plink_files)
    n = len(samples)

    if args.pheno_file is not None:
        pheno_path = args.pheno_file.expanduser().resolve()
        if not pheno_path.is_file():
            ap.error(f"--pheno-file not found: {pheno_path}")
        pheno_col = args.pheno_col
        LOGGER.info("Using external phenotype %s (column %s)", pheno_path, pheno_col)
    else:
        pheno_path = work_dir / "pheno.tsv"
        y = np.random.default_rng(args.input_seed).standard_normal(n)
        write_pheno(samples, y, pheno_path)
        pheno_col = "PHENO"
        LOGGER.info("Using synthetic phenotype (seed %d)", args.input_seed)

    inputs_dir = work_dir / "inputs"

    if args.no_rewrite_bim:
        bims = tuple(p.bim for p in plink_files)
        beds = tuple(p.bed for p in plink_files)
    else:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        bim_list: list[pathlib.Path] = []
        bed_list: list[pathlib.Path] = []
        for p in plink_files:
            new_bim, skipped = rewrite_bim(p.bim, inputs_dir / f"{p.bim.stem}.rewritten.bim")
            bim_list.append(new_bim)
            if skipped:
                total_rows = _count_bim_rows(p.bim)
                new_bed = write_filtered_bed(
                    p.bed, inputs_dir / f"{p.bim.stem}.filtered.bed",
                    n_individuals=n, skip_rows=set(skipped), total_rows=total_rows,
                )
                bed_list.append(new_bed)
            else:
                bed_list.append(p.bed)
        bims = tuple(bim_list)
        beds = tuple(bed_list)
    total_snps = sum(_count_bim_rows(b) for b in bims)

    t_prep = time.perf_counter() - t_prep0

    raw_stats_path = work_dir / f"bolt_stats_{uuid.uuid4().hex[:8]}.tsv"
    LOGGER.info("Raw BOLT stats will be written to %s (preserved)", raw_stats_path)

    t_run0 = time.perf_counter()
    run_official_bolt(
        bolt_bin=bolt_bin,
        fam=plink_files[0].fam,
        bims=bims,
        beds=beds,
        pheno=pheno_path,
        pheno_col=pheno_col,
        stats_file=raw_stats_path,
        num_threads=args.threads,
        total_snps=total_snps,
        num_leave_out_chunks=len(plink_files),
        bolt_seed=args.bolt_seed,
    )
    t_run = time.perf_counter() - t_run0

    shutil.copyfile(raw_stats_path, args.output)

    t_e2e = time.perf_counter() - t_e2e0

    # Scrape BOLT's own per-stage timings and summary stats from the log. Guard the
    # whole thing: a missing/reformatted log must not abort an otherwise-complete run.
    log_path = pathlib.Path(raw_stats_path).with_suffix(".log")
    bolt_staged: dict[str, float] = {}
    bolt_summary: dict = {}
    try:
        bolt_staged, bolt_summary = parse_bolt_log(log_path)
    except Exception as exc:  # noqa: BLE001 - best-effort log scrape
        LOGGER.warning("Failed to parse BOLT log %s: %s", log_path, exc)

    # End-to-end excluding the output-writing stage (still kept in bolt_staged).
    write_output_secs = bolt_staged.get(WRITE_OUTPUT_KEY, 0.0)
    total_elapsed = bolt_staged.get("total_elapsed")
    if total_elapsed is not None:
        bolt_staged["e2e_no_write"] = total_elapsed - write_output_secs

    LOGGER.info("prep:    %8.3f s", t_prep)
    LOGGER.info("run:     %8.3f s", t_run)
    LOGGER.info("e2e:     %8.3f s (python wall clock)", t_e2e)
    if total_elapsed is not None:
        LOGGER.info("bolt total elapsed:  %8.3f s", total_elapsed)
        LOGGER.info("bolt write output:   %8.3f s", write_output_secs)
        LOGGER.info("e2e_no_write:        %8.3f s (bolt total - write output)",
                    bolt_staged["e2e_no_write"])
    LOGGER.info("variants (BIM rows): %d", total_snps)
    LOGGER.info("Wrote %s", args.output)

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="bolt",
            params=vars(args),
            input_files=input_files_summary([t.bed for t in plink_files]),
            metrics={
                "py_metrics": {"prep": t_prep, "run": t_run, "e2e": t_e2e},
                "bolt_staged": bolt_staged,
                "e2e": t_e2e
            },
            output_summary={
                "variants": total_snps,
                "output": str(args.output),
                "h2": bolt_summary.get("h2"),
                "sigma2_K": bolt_summary.get("sigma2_K"),
                "calibration": bolt_summary.get("calibration"),
            },
        )


if __name__ == "__main__":
    main()
