# Pure cuSPARSE SpTrSV for the GRG triangular operator

The GRG linear operator's UP traversal computes, in stable-height (topological)
order:

```
node_values = b + A @ node_values        # A strictly block-lower-triangular
```

which is algebraically the triangular solve `(I − A) x = b`. Instead of
emulating that solve with a chain of SpMVs (as `grg-spmv/` does), this directory
materializes `M = I − A` once and solves it with cuSPARSE's dedicated, highly
optimized sparse triangular solve (`cusparseSpSV` / `cusparseSpSM`).

## Files

- `grg_to_csr.py` — reads a `.grg` with `grgl`/`pygrgl`, reorders nodes into the
  same stable-height order the SpMM compiler uses, and writes `M = I − A` as a
  CSR matrix to a compact binary `.csr` file.
- `sptrsv.cu` — standalone CUDA benchmark: loads the `.csr`, solves `op(M) x = b`
  with cuSPARSE, times the analysis and solve phases, and prints a residual
  sanity check.
- `Makefile` — toolkit-agnostic build (`nvcc` from `PATH`, `CUDA_HOME`/`ARCH`
  overridable).

## Binary `.csr` format (little-endian)

```
magic        : 8 bytes  = "GRGTRSV1"
index_bytes  : int32    = 4 (int32 col indices) or 8 (int64 col indices)
nrows        : int64    (= num_nodes)
ncols        : int64    (= num_nodes)
nnz          : int64
indptr       : (nrows+1) x int64
indices      : nnz x (index_bytes)
values       : nnz x float64
```

`M` is lower triangular with an explicitly stored unit diagonal, so
`nnz == num_nodes + num_edges`. Column indices are sorted within each row
(required by cuSPARSE). Values are stored as `float64`; the benchmark casts to
`float` or `double` on load.

## Build

`nvcc` (CUDA 12.x or 13.x) and cuSPARSE must be available. The generic
SpSV/SpSM API used here is stable across both.

```
make                       # nvcc on PATH, default ARCH=sm_80 (A100)
make ARCH=sm_90            # Hopper (H100)
make CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9   # explicit toolkit
```

## Run

```
# 1. Convert a .grg into the CSR matrix M = I - A
/global/homes/y/yfli03/.conda/envs/trsv_grg/bin/python grg_to_csr.py \
    /global/homes/y/yfli03/grg/grgl/jupyter/simple_example.grg M.csr

# 2. Build the benchmark
make

# 3. Solve / benchmark
#    sptrsv <matrix.csr> [--dtype float|double] [--k K] [--dir up|down] [--iters N]
#    defaults: --dtype double  --k 1  --dir up  --iters 10
./sptrsv M.csr --dtype double --k 1  --dir up
./sptrsv M.csr --dtype float  --k 16 --dir down
```

### Options

| flag       | meaning                                                              | default  |
|------------|----------------------------------------------------------------------|----------|
| `--dtype`  | compute precision: `float` or `double`                               | `double` |
| `--k`      | number of dense RHS columns; `K=1` → `cusparseSpSV`, `K>1` → `SpSM`  | `1`      |
| `--dir`    | `up` solves `M x = b`; `down` solves `Mᵀ x = b` (transpose op)       | `up`     |
| `--iters`  | timed solve repetitions (reported time is the mean)                  | `10`     |

The RHS is filled with ones (benchmark only). After solving, the program prints
the analysis time, mean solve time, achieved GFLOP/s, and the relative residual
`‖op(M) x − b‖ / ‖b‖`, which should be ~`1e-14` for `double` and ~`1e-6` for
`float`.

**Timed region.** For a fair comparison against the SpMM baseline (whose timed
`matmul` includes on-GPU input/output handling), each timed iteration wraps the
`cusparseSpSV`/`SpSM` solve with a same-GPU **device-to-device** copy of the input
(RHS → solver buffer) and of the output (solver result → downstream sink). The
reported `solve time` is the mean over `--iters` of `copy-in + solve + copy-out`.
The one-time `analysis` phase is timed separately and excluded.

## Benchmark harness integration

`sptrsv` is wired into the top-level harness as the `kernel`/`trsv` backend via
`examples/kernel/trsv.py`, which runs the binary on one `.csr` and appends a
unified JSON-Lines record. The binary prints a machine-readable `RESULT_JSON:`
line (per-trial `solve_trials_ms`, `analysis_ms`, `buffer_bytes`, `residual`)
that the wrapper parses. Run it through:

```
python evaluate.py -a kernel -b trsv -d /path/to/M.csr \
    --runs 3 --record records/kernel_trsv.jsonl -- --direction up --k 4 --dtype float64
```

## Notes

- For correctness the matrix's stable-height ordering matches
  `grg-spmv/pygrgl_spmv/grg/compile.py` (`_compute_node_levels`,
  `_build_stable_height_order`), which `grg_to_csr.py` imports and reuses.
- DOWN reuses the *same* lower-triangular matrix via
  `CUSPARSE_OPERATION_TRANSPOSE`; no separate upper-triangular matrix is stored.
