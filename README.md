## Mikado (Preprint Artifacts)
This repository collects the code artifacts for the MIKADO preprint [link].
The goal of this repo is to provide the artifacts necessary for reproducing the results presented in the paper.
Code Authors: Qingyao Sun, Yifan Li


## Manual Installation

Manual installation is supported for GRGL, MIKADO(CPU), and MIKADO(GPU) runs. It is not currently supported for the graph-first implementation and the cuSparse SpSV approach.

GRGL, Grapp, and GRG-SpMV are included as git submodules, so clone this repository recursively:
```
git clone --recurse-submodules git@github.com:CornellHPC/mikado.git
```

If you already cloned without --recurse-submodules, the `grgl/`, `grapp/`, and `grg-spmv/` directories will be empty. Populate them with:

```
git submodule update --init --recursive
```

Install the three repositories in this order, following the setup instructions in each repository's own README:
  1. **GRGL** (`grgl/`) — provides `pygrgl`, which includes GRGL backend, and GRG construction and manipulation-related functionalities.
  2. **Grapp** (`grapp/`) — provides the GRG-based application implementations.
  3. **GRG-SpMV** (`grg-spmv/`) — provides the core implementation of MIKADO, including both CPU and GPU backends.

## Using Dockerfiles

There are two docker files provided under `docker_images/`. 
- `mikado` is the main image for running MIKADO experiments.
- `graph-first` is the image for the graph-first implementation.
Docker images can be built using the provided Dockerfiles.

## Running Experiments

In side the `mikado` repo, benchmark scripts are provided under `benchmark`. The single entry point for user is `evaluate.py`, and the application and backend specific scripts are under `benchmark/examples`.

These parameters apply to all calls to `evaluate.py`: 
- `-a APPLICATION`  benchmark application (one of: bolt, gwas, kernel, pca, pca_lobpcg)
- `-b BACKEND` backend implementation, validated against the chosen application. grgl, mkl, and cusparse are available for every application; boltlmm only for bolt; plink only for pca and gwas; trsv and legacy only for kernel. pca_lobpcg has no baseline backend.
- `-d DATASET` dataset path, forwarded to the backend via its dataset adapter
- `--warmup WARMUP` number of warmup runs
- `--runs RUNS` number of experiment runs (each run will be a line of record)
- `--record RECORD` record file (JSONL). Each line will contain the parameters, basic system info, and results of one experiment run.
- `--output OUTPUT` result file (TSV). Available for all applications except kernel.
- `--skip-output` skip output writing for GWAS, BOLT-LMM, PCA runs.

There're some additional backend-specific parameters. The MKL and cuSparse ones are `evaluate.py`'s own parameters. Some parameters have to be passed after `--`, which means they're passed to the underlying application/backend-specific script directly. Such parameters will be noted.

For MIKADO CPU (MKL):
- `--mkl-threads <N>` number of threads to use for MKL. This can be a single number, or a map, e.g. `benchmark/configs/mkl_threads_d32.json`, which allocate different number of threads to different chromosomes
- `--optimize` enable MKL optimization flags. This will consume a lot more memory and need some optimization time. After optimization MKL may run faster.

For MIKADO GPU (cuSparse):
- `--native`, `--no-native` defaults to on. In native mode the backend uses CuPy arrays, GPU-to-GPU communication for multiple chromosome runs, and CuPy's PCA. Better performance.
- `--capture`, `--no-capture` defaults to on. Use CUDA graph capture. Better performance.
- `--device-map <FILE>` device map file for the placement of chromosomes on GPUs, e.g. `benchmark/configs/device_map_[1/2/4].json`. Always use `device_map_1.json` when only a single chromosome is used.
- `--force-spmm`, `--no-force-spmm` defaults to off. When enabled, use k=2 even when computing k=1 cases, to remedy the cuSparse precision error. **For CUDA >= 13.3.1 it should be turned off.**
- `--tol-record <FILE>` a GRGL / MIKADO CPU / MIKADO GPU non-native run record. Pass it when running PCA with cuSparse in native mode. SciPy uses relative error as the stopping condition and CuPy uses absolute, so CuPy's threshold is taken from the other run's result to keep the comparison fair.

For PLINK2, **all extra parameters need to be passed after `--`**:
- `--plink2-bin <PATH>` path to the plink2 binary, e.g. `/usr/local/bin/plink2`. Required.
- `--chromosomes <list>` comma-separated chromosomes to merge and analyze, e.g. `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22`. Required for pca only.
- `--threads <N>` plink2's own `--threads`

For (Original) BOLT-LMM, **all extra parameters need to be passed after `--`**:
- `--bolt-bin <PATH>` path to the BOLT-LMM binary, e.g. `/opt/BOLT-LMM_v2.5/bolt`. Required.
- `--chromosomes 19,20,21,22` comma-separated chromosomes to analyze, matching the dataset. Required.
- `--threads <N>` number of threads BOLT-LMM uses.

There're some additional application-specific parameters:

For application bolt:
- `--bolt-pheno-file <FILE>` phenotype file, e.g. `<dataset>/pheno.txt`. **Always pass it, otherwise a synthetic seeded phenotype is used.** 

For kernel, **these need to be passed after `--`**:
- `--direction up|down` matmul direction. Defaults to up.
- `--k <N>` number of probe vectors. k=1 is SpMV, k>1 is SpMM. Defaults to 1.

For pca_lobpcg, **after `--`**:
- `--pcs <N>` number of eigenvectors to compute. Defaults to 10.

