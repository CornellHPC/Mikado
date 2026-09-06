// Standalone cuSPARSE SpTrSV benchmark for the GRG triangular operator.
//
// Loads the lower-triangular matrix M = I - A produced by grg_to_csr.py (in
// stable-height / topological order) and solves M x = b (UP, non-transpose) or
// M^T x = b (DOWN, transpose) with cuSPARSE's dedicated sparse triangular solve.
//
//   K == 1 : cusparseSpSV  (single dense vector)
//   K  > 1 : cusparseSpSM  (K dense right-hand sides, column-major)
//
// The compute datatype (float/double), number of RHS columns K, direction, and
// number of timed solve iterations are all selectable on the command line.
//
//   sptrsv <matrix.csr> [--dtype float|double] [--k K] [--dir up|down] [--iters N]
//
// Defaults: --dtype double  --k 1  --dir up  --iters 10
//
// Benchmark only: the RHS is filled with ones, analysis and solve phases are
// timed, and a host-side residual ||op(M) x - b|| / ||b|| is printed as a
// sanity check (should be ~machine-eps for double).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                   cudaGetErrorString(err__), __FILE__, __LINE__);             \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CHECK_CUSPARSE(call)                                                    \
  do {                                                                          \
    cusparseStatus_t st__ = (call);                                            \
    if (st__ != CUSPARSE_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuSPARSE error %s at %s:%d\n",                      \
                   cusparseGetErrorString(st__), __FILE__, __LINE__);          \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace {

constexpr char kMagic[8] = {'G', 'R', 'G', 'T', 'R', 'S', 'V', '1'};

struct CsrMatrix {
  int64_t nrows = 0;
  int64_t ncols = 0;
  int64_t nnz = 0;
  int index_bytes = 4;                 // 4 -> int32 indices, 8 -> int64
  // Widened-to-int64 host copies, used by the residual check.
  std::vector<int64_t> indptr;         // length nrows + 1
  std::vector<int64_t> col_index_i64;  // length nnz
  // Native-width raw copies for the H2D upload. cuSPARSE requires the row-offset
  // and column-index types to match, so both share index_bytes width.
  std::vector<int32_t> indptr_i32;
  std::vector<int32_t> colind_i32;
  std::vector<int64_t> indptr_i64;
  std::vector<int64_t> colind_i64;
  std::vector<double> values;          // file stores float64
};

CsrMatrix LoadCsr(const std::string& path) {
  FILE* fh = std::fopen(path.c_str(), "rb");
  if (!fh) {
    std::fprintf(stderr, "cannot open matrix file: %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }

  char magic[8];
  if (std::fread(magic, 1, 8, fh) != 8 || std::memcmp(magic, kMagic, 8) != 0) {
    std::fprintf(stderr, "bad magic in %s (expected GRGTRSV1)\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }

  CsrMatrix m;
  int32_t index_bytes = 0;
  if (std::fread(&index_bytes, sizeof(int32_t), 1, fh) != 1) std::exit(EXIT_FAILURE);
  m.index_bytes = index_bytes;
  if (index_bytes != 4 && index_bytes != 8) {
    std::fprintf(stderr, "unexpected index_bytes=%d\n", index_bytes);
    std::exit(EXIT_FAILURE);
  }
  if (std::fread(&m.nrows, sizeof(int64_t), 1, fh) != 1) std::exit(EXIT_FAILURE);
  if (std::fread(&m.ncols, sizeof(int64_t), 1, fh) != 1) std::exit(EXIT_FAILURE);
  if (std::fread(&m.nnz, sizeof(int64_t), 1, fh) != 1) std::exit(EXIT_FAILURE);

  const int64_t n_indptr = m.nrows + 1;
  m.indptr.resize(n_indptr);
  m.col_index_i64.resize(m.nnz);
  if (index_bytes == 4) {
    m.indptr_i32.resize(n_indptr);
    m.colind_i32.resize(m.nnz);
    if (std::fread(m.indptr_i32.data(), sizeof(int32_t), n_indptr, fh) !=
        static_cast<size_t>(n_indptr))
      std::exit(EXIT_FAILURE);
    if (std::fread(m.colind_i32.data(), sizeof(int32_t), m.nnz, fh) !=
        static_cast<size_t>(m.nnz))
      std::exit(EXIT_FAILURE);
    for (int64_t i = 0; i < n_indptr; ++i) m.indptr[i] = m.indptr_i32[i];
    for (int64_t i = 0; i < m.nnz; ++i) m.col_index_i64[i] = m.colind_i32[i];
  } else {
    m.indptr_i64.resize(n_indptr);
    m.colind_i64.resize(m.nnz);
    if (std::fread(m.indptr_i64.data(), sizeof(int64_t), n_indptr, fh) !=
        static_cast<size_t>(n_indptr))
      std::exit(EXIT_FAILURE);
    if (std::fread(m.colind_i64.data(), sizeof(int64_t), m.nnz, fh) !=
        static_cast<size_t>(m.nnz))
      std::exit(EXIT_FAILURE);
    for (int64_t i = 0; i < n_indptr; ++i) m.indptr[i] = m.indptr_i64[i];
    for (int64_t i = 0; i < m.nnz; ++i) m.col_index_i64[i] = m.colind_i64[i];
  }

  m.values.resize(m.nnz);
  if (std::fread(m.values.data(), sizeof(double), m.nnz, fh) !=
      static_cast<size_t>(m.nnz))
    std::exit(EXIT_FAILURE);

  std::fclose(fh);
  return m;
}

// Host CSR matvec used only for the residual sanity check.
// transpose=false: y = M x ; transpose=true: y = M^T x.
template <typename T>
void HostCsrMatvec(const CsrMatrix& m, bool transpose, const std::vector<T>& x,
                   std::vector<T>& y, int64_t k) {
  std::fill(y.begin(), y.end(), T(0));
  for (int64_t row = 0; row < m.nrows; ++row) {
    for (int64_t p = m.indptr[row]; p < m.indptr[row + 1]; ++p) {
      const int64_t col = m.col_index_i64[p];
      const T val = static_cast<T>(m.values[p]);
      for (int64_t c = 0; c < k; ++c) {
        // Column-major dense layout: element (i, c) at i + c*n.
        if (!transpose)
          y[row + c * m.nrows] += val * x[col + c * m.nrows];
        else
          y[col + c * m.nrows] += val * x[row + c * m.nrows];
      }
    }
  }
}

template <typename T>
double RelResidual(const CsrMatrix& m, bool transpose,
                   const std::vector<T>& x, const std::vector<T>& b, int64_t k) {
  std::vector<T> y(m.nrows * k);
  HostCsrMatvec<T>(m, transpose, x, y, k);
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    const double r = static_cast<double>(y[i]) - static_cast<double>(b[i]);
    num += r * r;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / (den > 0 ? std::sqrt(den) : 1.0);
}

template <typename T> cudaDataType CudaType();
template <> cudaDataType CudaType<float>() { return CUDA_R_32F; }
template <> cudaDataType CudaType<double>() { return CUDA_R_64F; }

struct Options {
  std::string matrix_path;
  std::string dtype = "double";
  int64_t k = 1;
  std::string dir = "up";
  int iters = 10;
  int warmup = 3;
};

template <typename T>
int Run(const CsrMatrix& m, const Options& opt) {
  const bool transpose = (opt.dir == "down");
  const cusparseOperation_t opA =
      transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cudaDataType value_type = CudaType<T>();
  // Row offsets and column indices share one index type (cuSPARSE requirement).
  const cusparseIndexType_t index_type =
      (m.index_bytes == 4) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
  const int64_t n = m.nrows;
  const int64_t k = opt.k;
  const T alpha = T(1);

  // ---- host buffers (compute precision) ----
  std::vector<T> h_values(m.nnz);
  for (int64_t i = 0; i < m.nnz; ++i) h_values[i] = static_cast<T>(m.values[i]);
  std::vector<T> h_b(n * k, T(1));   // RHS = ones
  std::vector<T> h_x(n * k, T(0));

  // ---- device buffers ----
  void* d_indptr = nullptr;
  void* d_colind = nullptr;
  void* d_values = nullptr;
  void* d_b = nullptr;       // RHS the solver reads
  void* d_x = nullptr;       // solution the solver writes
  void* d_b_src = nullptr;   // upstream input copied into d_b each iteration
  void* d_x_dst = nullptr;   // downstream sink the result is copied into
  const int64_t n_indptr = n + 1;
  CHECK_CUDA(cudaMalloc(&d_indptr, n_indptr * m.index_bytes));
  CHECK_CUDA(cudaMalloc(&d_colind, m.nnz * m.index_bytes));
  CHECK_CUDA(cudaMalloc(&d_values, m.nnz * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b, n * k * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x, n * k * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b_src, n * k * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_x_dst, n * k * sizeof(T)));

  if (m.index_bytes == 4) {
    CHECK_CUDA(cudaMemcpy(d_indptr, m.indptr_i32.data(),
                          n_indptr * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colind, m.colind_i32.data(),
                          m.nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMemcpy(d_indptr, m.indptr_i64.data(),
                          n_indptr * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colind, m.colind_i64.data(),
                          m.nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), m.nnz * sizeof(T),
                        cudaMemcpyHostToDevice));
  // The fair-timing input lives in d_b_src; each timed iteration copies it into
  // d_b (the buffer the solver reads) with a same-GPU device-to-device copy.
  CHECK_CUDA(cudaMemcpy(d_b_src, h_b.data(), n * k * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), n * k * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_x, 0, n * k * sizeof(T)));
  CHECK_CUDA(cudaMemset(d_x_dst, 0, n * k * sizeof(T)));

  cusparseHandle_t handle = nullptr;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Lower-triangular CSR with an explicitly stored unit diagonal.
  cusparseSpMatDescr_t matA = nullptr;
  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, n, m.ncols, m.nnz, d_indptr, d_colind, d_values,
      index_type, index_type, CUSPARSE_INDEX_BASE_ZERO, value_type));
  cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                           &fill, sizeof(fill)));
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                           &diag, sizeof(diag)));

  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));
  float analysis_ms = 0.0f;
  size_t buffer_bytes = 0;
  std::vector<float> trial_ms(opt.iters, 0.0f);  // per-iteration solve+copy time

  if (k == 1) {
    // -------- cusparseSpSV (single vector) --------
    cusparseDnVecDescr_t vecB = nullptr, vecX = nullptr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, n, d_b, value_type));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, value_type));

    cusparseSpSVDescr_t spsv = nullptr;
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsv));

    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        handle, opA, &alpha, matA, vecB, vecX, value_type,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv, &buffer_size));
    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
    buffer_bytes = buffer_size;
    std::printf("SpSV buffer  : %.3f MB\n", buffer_size / (1024.0 * 1024.0));

    CHECK_CUDA(cudaEventRecord(t0));
    CHECK_CUSPARSE(cusparseSpSV_analysis(handle, opA, &alpha, matA, vecB, vecX,
                                         value_type, CUSPARSE_SPSV_ALG_DEFAULT,
                                         spsv, d_buffer));
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&analysis_ms, t0, t1));

    // One iteration body: input copy (D2D) -> solve -> output copy (D2D).
    auto run_iter = [&]() {
      CHECK_CUDA(cudaMemcpyAsync(d_b, d_b_src, n * k * sizeof(T),
                                 cudaMemcpyDeviceToDevice));
      CHECK_CUSPARSE(cusparseSpSV_solve(handle, opA, &alpha, matA, vecB, vecX,
                                        value_type, CUSPARSE_SPSV_ALG_DEFAULT,
                                        spsv));
      CHECK_CUDA(cudaMemcpyAsync(d_x_dst, d_x, n * k * sizeof(T),
                                 cudaMemcpyDeviceToDevice));
    };
    for (int it = 0; it < opt.warmup; ++it) run_iter();
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int it = 0; it < opt.iters; ++it) {
      CHECK_CUDA(cudaEventRecord(t0));
      run_iter();
      CHECK_CUDA(cudaEventRecord(t1));
      CHECK_CUDA(cudaEventSynchronize(t1));
      CHECK_CUDA(cudaEventElapsedTime(&trial_ms[it], t0, t1));
    }

    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsv));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
  } else {
    // -------- cusparseSpSM (K dense RHS, column-major) --------
    cusparseDnMatDescr_t matB = nullptr, matX = nullptr;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, n, k, n, d_b, value_type,
                                       CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matX, n, k, n, d_x, value_type,
                                       CUSPARSE_ORDER_COL));
    const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseSpSMDescr_t spsm = nullptr;
    CHECK_CUSPARSE(cusparseSpSM_createDescr(&spsm));

    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpSM_bufferSize(
        handle, opA, opB, &alpha, matA, matB, matX, value_type,
        CUSPARSE_SPSM_ALG_DEFAULT, spsm, &buffer_size));
    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
    buffer_bytes = buffer_size;
    std::printf("SpSM buffer  : %.3f MB\n", buffer_size / (1024.0 * 1024.0));

    CHECK_CUDA(cudaEventRecord(t0));
    CHECK_CUSPARSE(cusparseSpSM_analysis(handle, opA, opB, &alpha, matA, matB,
                                         matX, value_type,
                                         CUSPARSE_SPSM_ALG_DEFAULT, spsm,
                                         d_buffer));
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&analysis_ms, t0, t1));

    // One iteration body: input copy (D2D) -> solve -> output copy (D2D).
    auto run_iter = [&]() {
      CHECK_CUDA(cudaMemcpyAsync(d_b, d_b_src, n * k * sizeof(T),
                                 cudaMemcpyDeviceToDevice));
      CHECK_CUSPARSE(cusparseSpSM_solve(handle, opA, opB, &alpha, matA, matB,
                                        matX, value_type,
                                        CUSPARSE_SPSM_ALG_DEFAULT, spsm));
      CHECK_CUDA(cudaMemcpyAsync(d_x_dst, d_x, n * k * sizeof(T),
                                 cudaMemcpyDeviceToDevice));
    };
    for (int it = 0; it < opt.warmup; ++it) run_iter();
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int it = 0; it < opt.iters; ++it) {
      CHECK_CUDA(cudaEventRecord(t0));
      run_iter();
      CHECK_CUDA(cudaEventRecord(t1));
      CHECK_CUDA(cudaEventSynchronize(t1));
      CHECK_CUDA(cudaEventElapsedTime(&trial_ms[it], t0, t1));
    }

    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUSPARSE(cusparseSpSM_destroyDescr(spsm));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matX));
  }

  CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, n * k * sizeof(T),
                        cudaMemcpyDeviceToHost));
  const double residual = RelResidual<T>(m, transpose, h_x, h_b, k);

  double solve_ms = 0.0;
  for (float v : trial_ms) solve_ms += v;
  solve_ms = trial_ms.empty() ? 0.0 : solve_ms / trial_ms.size();

  const double solve_s = solve_ms / 1e3;
  const double gflops = (solve_s > 0)
                            ? (2.0 * static_cast<double>(m.nnz) * k) / solve_s / 1e9
                            : 0.0;
  const char* routine = (k == 1) ? "cusparseSpSV" : "cusparseSpSM";

  std::printf("=== cuSPARSE SpTrSV benchmark ===\n");
  std::printf("matrix       : n=%lld nnz=%lld\n",
              static_cast<long long>(n), static_cast<long long>(m.nnz));
  std::printf("routine      : %s\n", routine);
  std::printf("dtype        : %s\n", opt.dtype.c_str());
  std::printf("K (rhs cols) : %lld\n", static_cast<long long>(k));
  std::printf("direction    : %s (%s)\n", opt.dir.c_str(),
              transpose ? "transpose" : "non-transpose");
  std::printf("analysis time: %.4f ms\n", analysis_ms);
  std::printf("solve time   : %.4f ms (mean of %d, warmup %d, incl. in/out D2D copies)\n",
              solve_ms, opt.iters, opt.warmup);
  std::printf("solve GFLOP/s: %.2f\n", gflops);
  std::printf("rel residual : %.3e ||op(M)x - b|| / ||b||\n", residual);

  // Machine-readable result line for the kernel/trsv.py wrapper.
  std::printf(
      "RESULT_JSON: {\"n\":%lld,\"nnz\":%lld,\"routine\":\"%s\","
      "\"dtype\":\"%s\",\"k\":%lld,\"direction\":\"%s\",\"warmup\":%d,"
      "\"iters\":%d,\"analysis_ms\":%.6f,\"buffer_bytes\":%llu,"
      "\"residual\":%.6e,\"solve_trials_ms\":[",
      static_cast<long long>(n), static_cast<long long>(m.nnz), routine,
      opt.dtype.c_str(), static_cast<long long>(k), opt.dir.c_str(), opt.warmup,
      opt.iters, analysis_ms, static_cast<unsigned long long>(buffer_bytes),
      residual);
  for (size_t i = 0; i < trial_ms.size(); ++i)
    std::printf("%s%.6f", (i ? "," : ""), trial_ms[i]);
  std::printf("]}\n");

  CHECK_CUDA(cudaEventDestroy(t0));
  CHECK_CUDA(cudaEventDestroy(t1));
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(d_indptr));
  CHECK_CUDA(cudaFree(d_colind));
  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_b_src));
  CHECK_CUDA(cudaFree(d_x_dst));
  return 0;
}

Options ParseArgs(int argc, char** argv) {
  Options opt;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (a == "--dtype") {
      opt.dtype = next("--dtype");
    } else if (a == "--k") {
      opt.k = std::stoll(next("--k"));
    } else if (a == "--dir") {
      opt.dir = next("--dir");
    } else if (a == "--iters") {
      opt.iters = std::stoi(next("--iters"));
    } else if (a == "--warmup") {
      opt.warmup = std::stoi(next("--warmup"));
    } else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: sptrsv <matrix.csr> [--dtype float|double] [--k K] "
          "[--dir up|down] [--iters N] [--warmup N]\n");
      std::exit(EXIT_SUCCESS);
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      std::exit(EXIT_FAILURE);
    } else {
      positional.push_back(a);
    }
  }
  if (positional.size() != 1) {
    std::fprintf(stderr,
                 "usage: sptrsv <matrix.csr> [--dtype float|double] [--k K] "
                 "[--dir up|down] [--iters N] [--warmup N]\n");
    std::exit(EXIT_FAILURE);
  }
  opt.matrix_path = positional[0];
  if (opt.dtype != "float" && opt.dtype != "double") {
    std::fprintf(stderr, "--dtype must be float or double\n");
    std::exit(EXIT_FAILURE);
  }
  if (opt.dir != "up" && opt.dir != "down") {
    std::fprintf(stderr, "--dir must be up or down\n");
    std::exit(EXIT_FAILURE);
  }
  if (opt.k < 1) {
    std::fprintf(stderr, "--k must be >= 1\n");
    std::exit(EXIT_FAILURE);
  }
  if (opt.iters < 1) opt.iters = 1;
  if (opt.warmup < 0) opt.warmup = 0;
  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt = ParseArgs(argc, argv);
  CsrMatrix m = LoadCsr(opt.matrix_path);
  if (opt.dtype == "float") return Run<float>(m, opt);
  return Run<double>(m, opt);
}
