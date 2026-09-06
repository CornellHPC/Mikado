# BOLT-LMM v2.5 reference build (source patch)

BOLT-LMM is the baseline we compare MIKADO's LMM path against. The precompiled
`bolt` binary shipped in the official v2.5 tarball **segfaults on modern glibc**:
its `__wrap_memcpy` compatibility shim recurses before `main` ever runs. So the
baseline is rebuilt from the bundled sources instead.

The build is kept faithful to the upstream Makefile's default configuration
(`-DUSE_MKL`, threaded MKL, `-O2 -msse2 -mavx`); only the compiler is changed
(`icpc` → `g++`). Nothing here is a performance tweak — the two source edits
below are compatibility fixes, so the baseline stays as close to the officially
distributed configuration as it can be on a current toolchain.

## Files

- `build_bolt.sh` — downloads the v2.5 tarball, applies the patches, builds with
  `g++` + threaded MKL, installs `bolt`, and preflights it by running `bolt -h`.
- `NonlinearOptMulti.cpp` — drop-in replacement for
  `BOLT-LMM_v2.5/src/NonlinearOptMulti.cpp` (see patch 1 below).

## Usage

```
MKLROOT=/opt/intel/oneapi/mkl/latest ./build_bolt.sh
```

The binary lands at `$BOLT_DIR/bolt` (default `/opt/BOLT-LMM_v2.5/bolt`); pass an
install directory as the first argument to put it elsewhere.

Environment overrides: `MKLROOT` (oneAPI MKL root), `BOLT_DIR` (unpack/install
location), `BOLT_URL` (source tarball), `JOBS` (parallel make jobs).

Build dependencies (Debian/Ubuntu package names): `build-essential`, `wget`,
`libboost-program-options-dev`, `libboost-iostreams-dev`, `libzstd-dev`,
`zlib1g-dev`, and `intel-oneapi-mkl-devel`. NLopt is **not** required — see
patch 1.

## Patches applied

1. **Stub the NLopt-dependent REML-AI path.** `NonlinearOptMulti.cpp` is the only
   consumer of NLopt in BOLT-LMM, and it implements the average-information REML
   path, which `--lmmInfOnly` never reaches. Replacing it with a throwing stub
   drops the NLopt dependency entirely while still failing loudly if that path is
   somehow exercised.
2. **Enable MKL** by adding `-DUSE_MKL` and MKL's include path to the Makefile.
   `-DUSE_MKL_MALLOC` is deliberately *not* set, per the official recommendation.
3. **Bump `-std=c++11` → `-std=c++14`.** Ubuntu 26.04's Boost.Math headers require
   C++14 (`std::is_final`, `*_t` trait aliases, `enable_if_t`), so C++11 no longer
   compiles. C++14 rather than C++17 keeps the deprecated features the old BOLT
   sources still rely on.

Linking uses the system Boost/zstd/zlib and `mkl_intel_lp64` + `mkl_gnu_thread` +
`mkl_core` with `libgomp` (the GNU OpenMP runtime, matching `g++`).

## Container build

`docker_images/grg-spmv/Dockerfile` invokes this script directly, from the
`mikado` checkout it clones into `/opt/mikado`:

```
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
RUN /opt/mikado/code/bolt_patch/build_bolt.sh
```

This directory is therefore the single source of truth for the BOLT baseline
build — `build_bolt.sh` must stay executable in git (`git update-index
--chmod=+x`) for that `RUN` to work.
