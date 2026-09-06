#!/usr/bin/env bash
#
# Build BOLT-LMM v2.5 from source with threaded MKL.
#
# The precompiled binary shipped in the BOLT-LMM v2.5 tarball segfaults on modern
# glibc (its __wrap_memcpy compat shim recurses before main), so we rebuild it.
# The build stays faithful to the upstream Makefile default config (-DUSE_MKL,
# threaded MKL, -O2 -msse2 -mavx), translated from icpc to g++. We do NOT set
# -DUSE_MKL_MALLOC (per the official recommendation), and the NLopt REML-AI path
# is stubbed out (unused by --lmmInfOnly), so NLopt is not needed.
#
# Usage:
#   MKLROOT=/opt/intel/oneapi/mkl/latest ./build_bolt.sh [install_dir]
#
# Environment:
#   MKLROOT    oneAPI MKL root (default: /opt/intel/oneapi/mkl/latest)
#   BOLT_DIR   where the tarball is unpacked (default: /opt/BOLT-LMM_v2.5)
#   BOLT_URL   source tarball URL
#   JOBS       parallel make jobs (default: nproc)
#
# The resulting binary is installed to "$BOLT_DIR/bolt", or to the directory
# given as the first argument (installed there as "<install_dir>/bolt").

set -euxo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"
BOLT_DIR="${BOLT_DIR:-/opt/BOLT-LMM_v2.5}"
BOLT_URL="${BOLT_URL:-https://storage.googleapis.com/broad-alkesgroup-public/BOLT-LMM/downloads/BOLT-LMM_v2.5.tar.gz}"
JOBS="${JOBS:-$(nproc)}"
INSTALL_DIR="${1:-$BOLT_DIR}"

[ -d "$MKLROOT" ] || { echo "MKLROOT=$MKLROOT does not exist" >&2; exit 1; }

# 0. Fetch and unpack upstream sources (skipped if already unpacked)
if [ ! -d "$BOLT_DIR/src" ]; then
  tmp_tar="$(mktemp -t bolt-XXXXXX.tar.gz)"
  wget -O "$tmp_tar" "$BOLT_URL"
  mkdir -p "$(dirname "$BOLT_DIR")"
  tar --no-same-owner -xzf "$tmp_tar" -C "$(dirname "$BOLT_DIR")"
  rm -f "$tmp_tar"
fi

cd "$BOLT_DIR/src"

# 1. Stub the NLopt-dependent REML-AI path so no NLopt symbols are referenced
cp "$PATCH_DIR/NonlinearOptMulti.cpp" NonlinearOptMulti.cpp

# 2. Enable MKL (USE_MKL only, NOT USE_MKL_MALLOC). Compiler/opt flags otherwise left at
#    the upstream defaults (-O2 -msse2 -mavx) on purpose.
grep -q 'DUSE_MKL' Makefile \
  || sed -i '/CFLAGS += -DUSE_SSE/a CFLAGS += -DUSE_MKL\nCPATHS += -I$(MKLROOT)/include' Makefile

# 2b. Bump -std=c++11 -> c++14 (compatibility, NOT optimization). Ubuntu 26.04's Boost.Math
#     headers require C++14 (std::is_final, *_t trait aliases, enable_if_t); c++11 fails to
#     compile. c++14 (not c++17) keeps the deprecated features old BOLT code may use.
sed -i 's/-std=c++11/-std=c++14/' Makefile

# 3. Build with g++ + threaded MKL; system boost/zstd; no NLopt
make -j "$JOBS" \
    CC=g++ \
    MKLROOT="${MKLROOT}" \
    BOOST_INSTALL_DIR= \
    NLOPT_INSTALL_DIR= \
    ZSTD_DIR=/usr/include \
    linking=dynamic \
    LLIBS="-lboost_program_options -lboost_iostreams -lzstd -lz" \
    LLAPACK="-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"

# 4. Install over the extracted-tree location and PREFLIGHT by actually running it
install -d "$INSTALL_DIR"
install -m 0755 bolt "$INSTALL_DIR/bolt"
"$INSTALL_DIR/bolt" -h > /dev/null
