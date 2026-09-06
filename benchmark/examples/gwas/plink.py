"""Standalone official plink2 --glm basic GWAS benchmark.

Self-contained: requires only stdlib + numpy + a plink2 binary (path supplied
via --plink2-bin). No imports from grg-spmv.

Drives plink2's ``--glm`` per-variant linear regression (genotype dosage +
intercept) over a single PLINK fileset -- BED/BIM/FAM (``--bfile``) or
PGEN/PVAR/PSAM (``--pfile``) -- the reference-tool analog of grapp's
``linear_assoc_no_covar`` / ``linear_assoc_covar``, so the grapp GWAS backends
can be benchmarked against it.

The input format is auto-detected from the given path: a .bed/.bim/.fam suffix
means bed, a .pgen/.pvar/.psam suffix means pgen, and a bare stem is probed for
.bed first, then .pgen. ``--format`` forces the choice.

Single fileset only (matching the single-GRG grapp gwas scripts). Phenotype: a
seeded standard-normal quantitative phenotype is generated from the .fam/.psam
(same RNG as the grapp scripts) unless --pheno-file is given. No covariates by
default (``allow-no-covars``); pass --covar-file to add plink-format covariates.
"""

import argparse
import logging
import os
import pathlib
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

import numpy

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary

LOGGER = logging.getLogger("plink_gwas")

# Per-format member suffixes: (genotypes, variants, samples).
_FORMAT_SUFFIXES = {
    "bed": (".bed", ".bim", ".fam"),
    "pgen": (".pgen", ".pvar", ".psam"),
}
# Which format a suffix belongs to (any member identifies the fileset).
_SUFFIX_FORMAT = {s: fmt for fmt, suffixes in _FORMAT_SUFFIXES.items() for s in suffixes}


@dataclass(frozen=True)
class PlinkFileset:
    """One PLINK fileset: its format, its three members, and their shared prefix."""

    fmt: str                # "bed" or "pgen"
    stem: pathlib.Path      # prefix shared by all three members
    geno: pathlib.Path      # .bed  | .pgen
    variants: pathlib.Path  # .bim  | .pvar
    samples: pathlib.Path   # .fam  | .psam

    @property
    def load_flag(self) -> str:
        """The plink2 flag that loads this fileset by stem."""
        return "--bfile" if self.fmt == "bed" else "--pfile"


def _fileset_from_stem(stem: pathlib.Path, fmt: str) -> PlinkFileset:
    """Build a PlinkFileset of the given format, verifying all three members exist."""
    geno_ext, var_ext, sample_ext = _FORMAT_SUFFIXES[fmt]
    members = {ext: stem.with_name(stem.name + ext)
               for ext in (geno_ext, var_ext, sample_ext)}
    for ext, path in members.items():
        if path.exists():
            continue
        # A zstd-compressed .pvar needs plink2's 'vzs' modifier and can't be counted
        # here, so say that outright rather than just reporting the .pvar as missing.
        if ext == ".pvar" and path.with_name(path.name + ".zst").exists():
            raise FileNotFoundError(
                f"{path} is missing; only the zstd-compressed {path.name}.zst is present. "
                f"Decompress it first: zstd -d {path}.zst"
            )
        raise FileNotFoundError(f"incomplete {fmt} fileset {stem}: {path} is missing")
    return PlinkFileset(fmt=fmt, stem=stem, geno=members[geno_ext],
                        variants=members[var_ext], samples=members[sample_ext])


def fileset_from_entry(entry, fmt: str = "auto") -> PlinkFileset:
    """Resolve a fileset path (any member, or the bare stem) into a PlinkFileset.

    A .bed/.bim/.fam suffix selects bed and a .pgen/.pvar/.psam suffix selects
    pgen; anything else is treated as a bare stem and probed for .bed first, then
    .pgen. ``fmt`` ("bed"/"pgen") forces the choice and must agree with an
    explicit suffix.
    """
    p = pathlib.Path(entry)
    suffix_fmt = _SUFFIX_FORMAT.get(p.suffix)
    if suffix_fmt is not None:
        if fmt != "auto" and fmt != suffix_fmt:
            raise ValueError(f"--format {fmt} contradicts the {p.suffix} input {p}")
        return _fileset_from_stem(p.with_suffix(""), suffix_fmt)

    # Bare stem: honour an explicit --format, else probe (bed wins ties).
    if fmt != "auto":
        return _fileset_from_stem(p, fmt)
    for candidate in ("bed", "pgen"):
        if p.with_name(p.name + _FORMAT_SUFFIXES[candidate][0]).exists():
            return _fileset_from_stem(p, candidate)
    raise FileNotFoundError(
        f"no PLINK fileset at stem {p}: neither {p.name}.bed nor {p.name}.pgen exists"
    )


def read_samples(fileset: PlinkFileset) -> list[tuple[str | None, str]]:
    """Return the (FID, IID) pairs from a fileset's .fam/.psam, in order.

    FID is None for a .psam whose header carries no FID column (plink2 allows an
    IID-only sample file).
    """
    path = fileset.samples
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines:
        raise ValueError(f"sample file is empty: {path}")

    # A .psam carries a '#'-prefixed header naming its columns; a .fam (or a
    # headerless .psam) is positional FID IID ...
    fid_col, iid_col, first_data = 0, 1, 0
    if lines[0].startswith("#"):
        header = lines[0].lstrip("#").split()
        if "IID" not in header:
            raise ValueError(f"{path}: header has no IID column: {lines[0].strip()}")
        iid_col = header.index("IID")
        fid_col = header.index("FID") if "FID" in header else None
        first_data = 1

    needed = iid_col if fid_col is None else max(fid_col, iid_col)
    samples: list[tuple[str | None, str]] = []
    for lineno, line in enumerate(lines[first_data:], start=first_data + 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) <= needed:
            raise ValueError(
                f"invalid sample line {lineno} in {path}: expected at least "
                f"{needed + 1} fields, got {len(fields)}"
            )
        samples.append((None if fid_col is None else fields[fid_col], fields[iid_col]))
    if not samples:
        raise ValueError(f"sample file has no samples: {path}")
    return samples


def count_variants(fileset: PlinkFileset) -> int:
    """Number of variants in a fileset's .bim/.pvar ('#' header lines don't count)."""
    with fileset.variants.open("rb") as f:
        return sum(1 for line in f if not line.startswith(b"#"))


def write_synthetic_pheno(samples, seed: int, out_path: pathlib.Path) -> pathlib.Path:
    """Write a seeded standard-normal quantitative phenotype (#FID IID PHENO1).

    Uses the same RNG as the grapp gwas scripts
    (``numpy.random.default_rng(seed).standard_normal(n)``) so the phenotype
    matches for a given seed and the runs are directly comparable. Samples
    without an FID (an IID-only .psam) get the two-column #IID PHENO1 form.
    """
    y = numpy.random.default_rng(seed).standard_normal(len(samples))
    has_fid = samples[0][0] is not None
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("#FID\tIID\tPHENO1\n" if has_fid else "#IID\tPHENO1\n")
        for (fid, iid), v in zip(samples, y):
            fh.write(f"{fid}\t{iid}\t{v:.6f}\n" if has_fid else f"{iid}\t{v:.6f}\n")
    return out_path


def _run_plink2(plink2_bin: pathlib.Path, cmd_args: list[str], log_path: pathlib.Path) -> None:
    """Run plink2 with the given args, tee stdout+stderr to log_path, raise on failure."""
    cmd = [str(plink2_bin), *cmd_args]
    LOGGER.info("Running plink2: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"plink2 failed with exit {result.returncode}; log: {log_path}\n"
            f"{result.stdout[-4000:]}"
        )


def run_glm(
    *,
    plink2_bin: pathlib.Path,
    fileset: PlinkFileset,
    pheno_file: pathlib.Path,
    covar_file: pathlib.Path | None,
    out_prefix: pathlib.Path,
    num_threads: int,
    memory_mb: int | None,
) -> pathlib.Path:
    """Run ``plink2 --bfile/--pfile <stem> --glm ...``; return the .glm.linear results path."""
    # allow-no-covars is required when running --glm without a covariate file;
    # with covariates it is dropped. hide-covar keeps just the additive genotype line.
    glm_args = ["hide-covar"]
    if covar_file is None:
        glm_args.insert(0, "allow-no-covars")
    cmd_args = [
        fileset.load_flag, str(fileset.stem),
        "--glm", *glm_args,
        "--pheno", str(pheno_file),
        "--threads", str(int(num_threads)),
        "--out", str(out_prefix),
    ]
    if covar_file is not None:
        cmd_args.extend(["--covar", str(covar_file)])
    if memory_mb is not None:
        cmd_args.extend(["--memory", str(int(memory_mb))])
    _run_plink2(plink2_bin, cmd_args, out_prefix.with_suffix(".log"))

    matches = sorted(out_prefix.parent.glob(f"{out_prefix.name}.*.glm.linear"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {out_prefix.name}.*.glm.linear output, found {len(matches)}"
        )
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",
                    help="A single PLINK fileset: a .bed/.bim/.fam or .pgen/.pvar/.psam path, "
                         "or the bare stem (the other two members must exist alongside it).")
    ap.add_argument("--format", choices=("auto", "bed", "pgen"), default="auto",
                    help="Input fileset format. Default: auto (from the input suffix; a bare "
                         "stem probes .bed first, then .pgen).")
    ap.add_argument("--plink2-bin", type=pathlib.Path, required=True,
                    help="Path to the plink2 binary.")
    ap.add_argument("--work-dir", type=pathlib.Path, required=True,
                    help="Working directory for the phenotype file and plink2 outputs (.glm.linear/.log).")
    ap.add_argument("--pheno-file", dest="pheno_file", type=pathlib.Path, default=None,
                    help="plink-format phenotype file (#FID IID PHENO...). Omitted -> a seeded "
                         "standard-normal phenotype is generated from the .fam/.psam (see --seed).")
    ap.add_argument("--covar-file", dest="covar_file", type=pathlib.Path, default=None,
                    help="plink-format covariate file (#FID IID COV...). Omitted -> no covariates "
                         "(--glm allow-no-covars).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the generated phenotype (used only when --pheno-file is omitted). Default: 0.")
    ap.add_argument("--threads", type=int, default=1,
                    help="plink2 --threads. Default: 1.")
    ap.add_argument("--memory", type=int, default=None,
                    help="plink2 --memory cap in MB (optional; plink2 auto-sizes if unset).")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("gwas_official_results.tsv"),
                    help="TSV path for the association table (copied from plink2's .glm.linear).")
    ap.add_argument("--skip-output", dest="skip_output", action="store_true",
                    help="Skip copying the .glm.linear TSV (timings/record still run).")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    plink2_bin = args.plink2_bin.expanduser().resolve()
    if not plink2_bin.is_file():
        ap.error(f"--plink2-bin not found: {plink2_bin}")
    if not os.access(plink2_bin, os.X_OK):
        ap.error(f"--plink2-bin is not executable: {plink2_bin}")

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    t_e2e0 = time.perf_counter()

    try:
        fileset = fileset_from_entry(args.input, args.format)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    LOGGER.info("PLINK input (%s): %s", fileset.fmt, fileset.geno)

    samples = read_samples(fileset)
    n_individuals = len(samples)
    total_variants = count_variants(fileset)
    LOGGER.info("%d individuals, %d variants", n_individuals, total_variants)

    # Phenotype: use the provided file or generate a seeded one.
    if args.pheno_file is not None:
        pheno_file = args.pheno_file.expanduser().resolve()
        if not pheno_file.is_file():
            ap.error(f"--pheno-file not found: {pheno_file}")
        LOGGER.info("Using phenotype file: %s", pheno_file)
    else:
        pheno_file = write_synthetic_pheno(samples, args.seed, work_dir / "pheno.txt")
        LOGGER.info("Generated seeded phenotype (seed=%d) -> %s", args.seed, pheno_file)

    covar_file = None
    if args.covar_file is not None:
        covar_file = args.covar_file.expanduser().resolve()
        if not covar_file.is_file():
            ap.error(f"--covar-file not found: {covar_file}")
        LOGGER.info("Using covariate file: %s", covar_file)

    out_prefix = work_dir / f"gwas_{uuid.uuid4().hex[:8]}"
    LOGGER.info("plink2 --glm output prefix: %s (preserved)", out_prefix)

    t_run0 = time.perf_counter()
    glm_path = run_glm(
        plink2_bin=plink2_bin,
        fileset=fileset,
        pheno_file=pheno_file,
        covar_file=covar_file,
        out_prefix=out_prefix,
        num_threads=args.threads,
        memory_mb=args.memory,
    )
    t_run = time.perf_counter() - t_run0

    if args.skip_output:
        LOGGER.info("Skipping output TSV (--skip-output)")
    else:
        shutil.copyfile(glm_path, args.output)
        LOGGER.info("Wrote %s (copied from %s)", args.output, glm_path.name)

    t_e2e = time.perf_counter() - t_e2e0

    LOGGER.info("run:     %8.3f s", t_run)
    LOGGER.info("e2e:     %8.3f s (python wall clock)", t_e2e)
    LOGGER.info("individuals: %d  variants: %d  covariates: %s",
                n_individuals, total_variants, "yes" if covar_file else "no")

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="gwas",
            params=vars(args),
            input_files=input_files_summary([fileset.geno]),
            metrics={"run": t_run, "e2e": t_e2e},
            output_summary={
                "num_individuals": int(n_individuals),
                "total_variants": int(total_variants),
                "input_format": fileset.fmt,
                "has_covar": covar_file is not None,
                "synthetic_pheno": args.pheno_file is None,
                "output": str(args.output),
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
