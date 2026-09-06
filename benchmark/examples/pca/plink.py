"""Standalone official plink2 --pca approx PCA benchmark.

Self-contained: only requires stdlib + a plink2 binary (path supplied via
--plink2-bin). No imports from grg-spmv.

Drives the official plink2 binary's randomized FastPCA (`--pca <k> approx`,
Galinsky et al. 2016) over the same PLINK inputs the grapp PCA scripts consume,
so grapp PCA can be benchmarked against the reference tool. Both PLINK fileset
formats are supported: BED/BIM/FAM (`--bfile`) and PGEN/PVAR/PSAM (`--pfile`).

The format is auto-detected -- a .bed/.bim/.fam path (or a chr<N>.bim in folder
mode) means bed, a .pgen/.pvar/.psam path (or a chr<N>.pvar) means pgen, and a
bare stem is probed for .bed first, then .pgen. `--format` forces the choice.
All filesets of one run must share a format.

Single fileset -> PCA runs directly on it. Multiple chromosomes -> the filesets
are first concatenated with `--pmerge-list` (positions are disjoint across
chromosomes, so the merge unions all variants over the shared sample set), then
PCA runs on the merged fileset. The merged fileset keeps the input format.

`approx` is always passed; it always uses mean-imputation for missing calls.
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

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _record import write_record, input_files_summary

LOGGER = logging.getLogger("plink_pca")

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


def _glob_chrom_variants(plink_dir: pathlib.Path, chrom: str, ext: str) -> tuple[pathlib.Path, ...]:
    """All chr<N> variant files (.bim/.pvar) in plink_dir matching the known layouts."""
    token = f"chr{chrom}"
    patterns = (
        plink_dir / token / f"{token}{ext}",
        plink_dir / f"{token}{ext}",
        plink_dir / f"{token}.*{ext}",
        plink_dir / f"{token}_*{ext}",
        plink_dir / f"{token}-*{ext}",
        plink_dir / f"*.{token}.*{ext}",
        plink_dir / f"*.{token}_*{ext}",
        plink_dir / f"*.{token}-*{ext}",
    )
    matches = []
    for pat in patterns:
        matches.extend(sorted(pat.parent.glob(pat.name)))
    return tuple(dict.fromkeys(p.resolve() for p in matches if p.exists()))


def _find_chrom_fileset(plink_dir: pathlib.Path, chrom: str, fmt: str) -> PlinkFileset:
    """Locate the single chr<N> fileset in plink_dir (bed first in auto mode)."""
    candidates = ("bed", "pgen") if fmt == "auto" else (fmt,)
    for candidate in candidates:
        ext = _FORMAT_SUFFIXES[candidate][1]
        unique = _glob_chrom_variants(plink_dir, chrom, ext)
        if not unique:
            continue
        if len(unique) != 1:
            raise FileNotFoundError(
                f"expected exactly one chr{chrom} {ext.upper()[1:]} in {plink_dir}, "
                f"found {len(unique)}"
            )
        return fileset_from_entry(unique[0], candidate)
    wanted = " or ".join(_FORMAT_SUFFIXES[c][1] for c in candidates)
    raise FileNotFoundError(f"no chr{chrom} {wanted} found in {plink_dir}")


def resolve_plink(args) -> tuple[PlinkFileset, ...]:
    """Resolve list mode (--plink) or folder mode (--plink-dir/--chromosomes) into filesets."""
    if args.plink is not None:
        filesets = tuple(fileset_from_entry(e, args.format) for e in args.plink)
    else:
        chroms = tuple(c.strip() for c in args.chromosomes.split(",") if c.strip())
        filesets = tuple(
            _find_chrom_fileset(args.plink_dir, c, args.format) for c in chroms
        )
    formats = {f.fmt for f in filesets}
    if len(formats) > 1:
        raise ValueError(
            "all input filesets must share one format, got "
            + ", ".join(f"{f.geno.name} ({f.fmt})" for f in filesets)
            + "; convert them or pass --format"
        )
    return filesets


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


def assert_same_samples(filesets) -> list[tuple[str | None, str]]:
    """Check every fileset lists the same samples in the same order; return them."""
    filesets = tuple(filesets)
    if not filesets:
        raise ValueError("at least one fileset is required")
    first = read_samples(filesets[0])
    for fileset in filesets[1:]:
        if read_samples(fileset) != first:
            raise ValueError(
                f"sample order differs between {filesets[0].samples} and {fileset.samples}"
            )
    return first


def count_variants(fileset: PlinkFileset) -> int:
    """Number of variants in a fileset's .bim/.pvar ('#' header lines don't count)."""
    with fileset.variants.open("rb") as f:
        return sum(1 for line in f if not line.startswith(b"#"))


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


def merge_filesets(
    *,
    plink2_bin: pathlib.Path,
    filesets: tuple[PlinkFileset, ...],
    work_dir: pathlib.Path,
    num_threads: int,
    memory_mb: int | None,
) -> PlinkFileset:
    """Concatenate multiple PLINK filesets via --pmerge-list; return the merged fileset.

    The merged fileset keeps the input format (bfile/--make-bed for bed,
    pfile/--make-pgen for pgen), so no conversion cost enters the merge timing.
    """
    fmt = filesets[0].fmt
    merge_list = work_dir / "merge_list.txt"
    with merge_list.open("w", encoding="utf-8") as fh:
        for f in filesets:
            fh.write(f"{f.stem}\n")

    merged_stem = work_dir / "merged"
    cmd_args = [
        "--pmerge-list", str(merge_list), "bfile" if fmt == "bed" else "pfile",
        "--make-bed" if fmt == "bed" else "--make-pgen",
        "--threads", str(int(num_threads)),
        "--out", str(merged_stem),
    ]
    if memory_mb is not None:
        cmd_args.extend(["--memory", str(int(memory_mb))])
    _run_plink2(plink2_bin, cmd_args, merged_stem.with_suffix(".log"))
    return _fileset_from_stem(merged_stem, fmt)


def run_pca(
    *,
    plink2_bin: pathlib.Path,
    fileset: PlinkFileset,
    pcs: int,
    out_prefix: pathlib.Path,
    num_threads: int,
    memory_mb: int | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Run `plink2 --bfile/--pfile <stem> --pca <k> approx`; return (.eigenvec, .eigenval)."""
    cmd_args = [
        fileset.load_flag, str(fileset.stem),
        "--pca", str(int(pcs)), "approx",
        "--threads", str(int(num_threads)),
        "--out", str(out_prefix),
    ]
    if memory_mb is not None:
        cmd_args.extend(["--memory", str(int(memory_mb))])
    _run_plink2(plink2_bin, cmd_args, out_prefix.with_suffix(".log"))

    eigenvec = out_prefix.with_suffix(".eigenvec")
    eigenval = out_prefix.with_suffix(".eigenval")
    for f in (eigenvec, eigenval):
        if not f.exists():
            raise RuntimeError(f"expected plink2 PCA output missing: {f}")
    return eigenvec, eigenval


def parse_eigenvals(path: pathlib.Path) -> list[float]:
    """Read a plink2 .eigenval file (one eigenvalue per line) into a list of floats."""
    vals: list[float] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                vals.append(float(line))
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plink", nargs="+",
                      help="List mode: PLINK .bim/.pvar paths (any fileset member, or stems). "
                           "The other two members must exist alongside each.")
    mode.add_argument("--plink-dir", type=pathlib.Path,
                      help="Folder mode: directory containing chr<N>.{bim,bed,fam} or "
                           "chr<N>.{pvar,pgen,psam}.")
    ap.add_argument("--chromosomes",
                    help="Comma-separated chromosomes (required with --plink-dir).")
    ap.add_argument("--format", choices=("auto", "bed", "pgen"), default="auto",
                    help="Input fileset format. Default: auto (from the input suffix; a bare "
                         "stem or a folder-mode search probes .bed first, then .pgen).")
    ap.add_argument("--plink2-bin", type=pathlib.Path, required=True,
                    help="Path to the plink2 binary.")
    ap.add_argument("--work-dir", type=pathlib.Path, required=True,
                    help="Working directory for intermediate files (merge list, merged fileset, eigen*/log).")
    ap.add_argument("-d", "--dimensions", "--pcs", dest="pcs", type=int, default=10,
                    help="Number of principal components to extract. Default: 10.")
    ap.add_argument("--threads", type=int, default=32,
                    help="plink2 --threads (merge and PCA). Default: 32.")
    ap.add_argument("--memory", type=int, default=None,
                    help="plink2 --memory cap in MB (optional; plink2 auto-sizes if unset).")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("pca_official_results.tsv"),
                    help="TSV path for the PC-score table (copied from plink2's .eigenvec).")
    ap.add_argument("--skip-output", dest="skip_output", action="store_true",
                    help="Skip copying the .eigenvec TSV (timings/record still run).")
    ap.add_argument("--record", type=pathlib.Path, default=None,
                    help="Append a JSON-Lines run record to this file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

    if args.plink_dir is not None and not args.chromosomes:
        ap.error("--chromosomes is required with --plink-dir")

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
        filesets = resolve_plink(args)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    input_format = filesets[0].fmt
    LOGGER.info("PLINK inputs (%d, %s):", len(filesets), input_format)
    for f in filesets:
        LOGGER.info("  %s", f.geno)

    samples = assert_same_samples(filesets)
    n_individuals = len(samples)
    total_variants = sum(count_variants(f) for f in filesets)
    LOGGER.info("%d individuals, %d total variants across %d fileset(s)",
                n_individuals, total_variants, len(filesets))

    # Merge step (only needed for multiple filesets).
    t_merge = 0.0
    if len(filesets) == 1:
        pca_input = filesets[0]
    else:
        t_merge0 = time.perf_counter()
        pca_input = merge_filesets(
            plink2_bin=plink2_bin,
            filesets=filesets,
            work_dir=work_dir,
            num_threads=args.threads,
            memory_mb=args.memory,
        )
        t_merge = time.perf_counter() - t_merge0
        LOGGER.info("Merged %d filesets -> %s", len(filesets), pca_input.stem)

    # PCA step.
    out_prefix = work_dir / f"pca_{uuid.uuid4().hex[:8]}"
    LOGGER.info("plink2 PCA output prefix: %s (preserved)", out_prefix)

    t_run0 = time.perf_counter()
    eigenvec, eigenval = run_pca(
        plink2_bin=plink2_bin,
        fileset=pca_input,
        pcs=args.pcs,
        out_prefix=out_prefix,
        num_threads=args.threads,
        memory_mb=args.memory,
    )
    t_run = time.perf_counter() - t_run0

    eig_vals = parse_eigenvals(eigenval)

    if args.skip_output:
        LOGGER.info("Skipping output TSV (--skip-output)")
    else:
        shutil.copyfile(eigenvec, args.output)
        LOGGER.info("Wrote %s (copied from %s)", args.output, eigenvec.name)

    t_e2e = time.perf_counter() - t_e2e0

    LOGGER.info("merge:   %8.3f s", t_merge)
    LOGGER.info("run:     %8.3f s", t_run)
    LOGGER.info("e2e:     %8.3f s (python wall clock)", t_e2e)
    LOGGER.info("individuals: %d  variants: %d  PCs: %d", n_individuals, total_variants, args.pcs)
    LOGGER.info("eigenvalues: %s", ", ".join(f"{v:.4g}" for v in eig_vals))

    if args.record:
        write_record(
            args.record,
            script=pathlib.Path(__file__).name,
            application="pca",
            params=vars(args),
            input_files=input_files_summary([f.geno for f in filesets]),
            metrics={"merge": t_merge, "run": t_run, "e2e": t_e2e},
            output_summary={
                "num_individuals": int(n_individuals),
                "num_pcs": int(args.pcs),
                "num_filesets": len(filesets),
                "total_variants": int(total_variants),
                "input_format": input_format,
                "eig_vals": eig_vals,
                "approx": True,
                "output": str(args.output),
            },
        )
        LOGGER.info("Appended run record to %s", args.record)


if __name__ == "__main__":
    main()
