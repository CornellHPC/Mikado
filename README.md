<p align="center">
  <img src="mikado-lockup-transparent-3600.png" alt="Mikado" width="480">
</p>

# Mikado — Preprint Artifacts

This repository collects the code artifacts for the Mikado preprint, *Sparse Linear Algebra Accelerates Genotype Representation Graph Computation at Biobank Scale* ([link TBD](https://github.com/CornellHPC/Mikado/edit/preprint/README.md)). It provides everything needed to reproduce the results presented in the paper.

## Table of Contents

- [Citation](#citation)
- [Installation](#installation)
  - [Manual Installation](#manual-installation)
  - [Using Dockerfiles](#using-dockerfiles)
- [Running Experiments](#running-experiments)
  - [Converting a Dataset](#converting-a-dataset)
  - [Running Parameters](#running-parameters)
  - [Backend-Specific Parameters](#backend-specific-parameters)
  - [Application-Specific Parameters](#application-specific-parameters)
  - [Examples](#examples)
  - [Phenotype File](#phenotype-file)
- [Acknowledgements](#acknowledgements)

## Citation

If you use Mikado in your work (to appear online by 9/15/2026), please cite:

```bibtex
@misc{mikado,
  title         = {Sparse Linear Algebra Accelerates Genotype Representation Graph Computation at Biobank Scale},
  author        = {Li, Yifan and Sun, Qingyao and DeHaas, Drew and Zhao, Max Xiaohang and Boyko, Adam R. and Musharoff, Shaila A. and Wei, Xinzhu and Guidi, Giulia},
  year          = {2026},
  eprint        = {<arxiv-id>},
  archivePrefix = {arXiv},
  primaryClass  = {<primary-class>},
  url           = {<preprint-url>}
}
```

## Installation

### Manual Installation

Manual installation is supported for GRGL, Mikado (CPU), and Mikado (GPU) runs. It is not currently supported for the graph-first implementation or the cuSparse SpSV approach.

GRGL, Grapp, and GRG-SpMV are included as git submodules, so clone this repository recursively:

```bash
git clone --recurse-submodules git@github.com:CornellHPC/mikado.git
```

If you already cloned without `--recurse-submodules`, the `grgl/`, `grapp/`, and `grg-spmv/` directories will be empty. Populate them with:

```bash
git submodule update --init --recursive
```

Install the three repositories in the following order, following the setup instructions in each repository's own README:

1. **GRGL** (`grgl/`) — provides `pygrgl`, which includes the GRGL backend and GRG construction and manipulation functionality.
2. **Grapp** (`grapp/`) — provides the GRG-based application implementations.
3. **GRG-SpMV** (`grg-spmv/`) — provides the core implementation of Mikado, including both CPU and GPU backends.

### Using Dockerfiles

Two Dockerfiles are provided under `docker_images/`:

- `mikado` — the main image for running Mikado experiments.
- `graph-first` — the image for the graph-first implementation.

Build either image using the provided Dockerfile.

## Running Experiments

Benchmark scripts live under `benchmark/`. The single entry point is `evaluate.py`; application- and backend-specific scripts are under `benchmark/examples/`.

### Converting a Dataset

To construct a `.grg` file from formats such as `.vcf.gz`, refer to the [GRGL docs](https://grgl.readthedocs.io/en/stable/).

The `.grg` file then needs a simple conversion step to generate a `.grg_spmv` artifact, the format that `pygrgl-spmv` consumes. Use the `simple_convert` function provided in the GRG-SpMV library:

```python
from pygrgl_spmv import simple_convert

artifact = simple_convert("chr1.grg", "artifacts/chr1.grg_spmv")
```

Multi-processing can speed up conversion of multiple datasets.

### Running Parameters

The following parameters apply to all calls to `evaluate.py`:

| Parameter | Description |
| --- | --- |
| `-a APPLICATION` | Benchmark application: one of `bolt`, `gwas`, `kernel`, `pca`, `pca_lobpcg`. |
| `-b BACKEND` | Backend implementation, validated against the chosen application. `grgl`, `mkl`, and `cusparse` are available for every application; `boltlmm` only for `bolt`; `plink` only for `pca` and `gwas`; `trsv` and `legacy` only for `kernel`. `pca_lobpcg` has no baseline backend. |
| `-d DATASET` | Dataset path, forwarded to the backend via its dataset adapter. Either a file (one chromosome) or a directory of files (multiple chromosomes), depending on application and backend. |
| `--warmup WARMUP` | Number of warmup runs. |
| `--runs RUNS` | Number of experiment runs (each run is one record line). |
| `--record RECORD` | Record file (JSONL). Each line holds the parameters, basic system info, and results of one run. |
| `--output OUTPUT` | Result file (TSV). Available for all applications except `kernel`. |
| `--skip-output` | Skip output writing for GWAS, BOLT-LMM, and PCA runs. |
| `--work-dir DIR` | Directory for intermediate files (PLINK2 and original BOLT-LMM only). |

### Backend-Specific Parameters

Some parameters are `evaluate.py`'s own (MKL and cuSparse); others must be passed after `--`, meaning they go directly to the underlying application- or backend-specific script. Parameters requiring `--` are noted below.

**Mikado CPU (MKL)**

| Parameter | Description |
| --- | --- |
| `--mkl-threads <N>` | Number of MKL threads. A single number, or a map (e.g. `benchmark/configs/mkl_threads_d32.json`) that allocates different thread counts per chromosome. |
| `--optimize` | Enable MKL optimization flags. Consumes more memory and adds optimization time, but MKL may run faster afterward. |

**Mikado GPU (cuSparse)**

| Parameter | Description |
| --- | --- |
| `--native`, `--no-native` | Defaults to on. Native mode uses CuPy arrays, GPU-to-GPU communication for multi-chromosome runs, and CuPy's PCA, for better performance. |
| `--capture`, `--no-capture` | Defaults to on. Use CUDA graph capture for better performance. |
| `--device-map <FILE>` | Device map for chromosome placement across GPUs (e.g. `benchmark/configs/device_map_[1/2/4].json`). Controls how many GPUs are used when multiple are available. Device 0 is always used for non-GRG computation. Use `device_map_1.json` for single-chromosome runs. |
| `--force-spmm`, `--no-force-spmm` | Defaults to off. When enabled, uses k=2 even for k=1 cases to remedy the cuSparse precision error. **Turn off for CUDA >= 13.3.1.** |
| `--tol-record <FILE>` | A GRGL / Mikado CPU / Mikado GPU non-native run record. Pass it when running PCA with cuSparse in native mode: SciPy uses relative error as its stopping condition while CuPy uses absolute, so CuPy's threshold is taken from the other run's result to keep the comparison fair. |

**PLINK2** (all extra parameters after `--`)

| Parameter | Description |
| --- | --- |
| `--plink2-bin <PATH>` | Path to the plink2 binary (e.g. `/usr/local/bin/plink2`). Required. |
| `--chromosomes <list>` | Comma-separated chromosomes to merge and analyze (e.g. `1,2,...,22`). Required for `pca` only. |
| `--threads <N>` | plink2's own `--threads`. |

**Original BOLT-LMM** (all extra parameters after `--`)

| Parameter | Description |
| --- | --- |
| `--bolt-bin <PATH>` | Path to the BOLT-LMM binary (e.g. `/opt/BOLT-LMM_v2.5/bolt`). Required. |
| `--chromosomes <list>` | Comma-separated chromosomes to analyze, matching the dataset (e.g. `19,20,21,22`). Required. |
| `--threads <N>` | Number of threads BOLT-LMM uses. |

### Application-Specific Parameters

**bolt**

| Parameter | Description |
| --- | --- |
| `--bolt-pheno-file <FILE>` | Phenotype file (e.g. `<dataset>/pheno.txt`). **Always pass it; otherwise a synthetic seeded phenotype is used.** See [Phenotype File](#phenotype-file) for format. |

**kernel** (after `--`)

| Parameter | Description |
| --- | --- |
| `--direction up\|down` | Matmul direction. Defaults to `up`. |
| `--k <N>` | Number of probe vectors. k=1 is SpMV, k>1 is SpMM. Defaults to 1. |

**pca_lobpcg** (after `--`)

| Parameter | Description |
| --- | --- |
| `--pcs <N>` | Number of eigenvectors to compute. Defaults to 10. |

### Examples

The examples below assume a Docker instance where the simulated 200k dataset is mounted at `/sim2/`, the 1000 Genomes dataset at `/1kg/`, and the output records directory at `/records/`.

**Kernel — Mikado CPU (MKL)** (requires `.grg_spmv` files):

```bash
cd /opt/mikado/benchmark
source /opt/intel/oneapi/mkl/latest/env/vars.sh
python3 evaluate.py -a kernel -b mkl \
  -d /sim2/chr11.grg_spmv \
  --mkl-threads 16 --record /records/mkl_ker.jsonl \
  -- --direction up --k 1
```

> The `source` command is necessary for MKL executions in Docker.

**GWAS — PLINK2** (requires `.bed`/`.bim`/`.fam` files):

```bash
cd /opt/mikado/benchmark
python3 evaluate.py -a gwas -b plink -d /sim2/chr21.bed \
  --warmup 1 --runs 3 \
  --work-dir /tmp \
  --record /records/gwas_plink.jsonl \
  -- --plink2-bin /usr/local/bin/plink2 --threads 32
```

**PCA — GRGL** (requires `.grg` files):

```bash
cd /opt/mikado/benchmark
python3 evaluate.py -a pca -b grgl \
  -d /sim2/chr11.grg --warmup 1 --runs 3 \
  --record /records/grgl_pca.jsonl
```

**BOLT-LMM — Mikado GPU (cuSparse), 4 devices** (requires `.grg_spmv` files):

```bash
cd /opt/mikado/benchmark
python3 evaluate.py -a bolt -b cusparse -d /1kg/ \
  --runs 1 --capture --native --no-force-spmm \
  --device-map /opt/mikado/benchmark/configs/device_map_4.json \
  --bolt-pheno-file /1kg/pheno.txt \
  --record /records/bolt_cus.jsonl \
  --output /1kg/bolt_1kg_cus.tsv
```

> A phenotype file (`/1kg/pheno.txt` here) is required for BOLT-LMM runs.

### Phenotype File

Following the standard format, the `FID IID PHENO` layout is recommended:

```
FID IID PHENO
0 HG00096 -685.90926499272052
0 HG00097 -754.09346295775549
0 HG00099 -620.97794377676041
0 HG00100 -465.04867997694726
0 HG00101 -546.96799947704483
0 HG00102 -466.26337135287901
0 HG00103 -623.33471631632881
0 HG00105 -670.1759539529744
0 HG00106 -746.82045085437449
```

For Grapp-based approaches (including Mikado), phenotype rows are matched to individuals **by position, not by ID**: the FID and IID columns are ignored entirely, and only the last column is read. Row i must therefore correspond to GRG individual i.

GRG preserves individual order at construction, and `.grg_spmv` files preserve that order, so alignment holds by default. Manual checking is nonetheless recommended when in doubt. The snippet below prints `FID_IID` in individual order for a `.grg` file:

```python
import pygrgl

grg = pygrgl.load_immutable_grg(str(PATH_TO_GRG), load_up_edges=False)
n = int(grg.num_individuals)
for i in range(n):
    print(grg.get_individual_id(i))
```

Input covariates must be ordered accordingly.

## Acknowledgements

The authors gratefully acknowledge *All of Us* participants for their contributions, without whom this research would not have been possible. We thank Andrew Clark and Can Firtina for their feedback on the manuscript. In addition, we thank the National Institutes of Health *All of Us* Research Program for making available the participant data examined in this study. This study used data from the *All of Us* Research Program Controlled Tier Dataset CDRv8, available to authorized users on the Researcher Workbench.

This material is based upon work supported by the National Science Foundation under Grant IIS-2435801. This research used resources from the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231, using NERSC award ASCR-ERCAP0030076. This work used DeltaAI at the National Center for Supercomputing Applications (NCSA) through allocation CIS251351 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.
