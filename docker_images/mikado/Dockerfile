FROM docker.io/nvidia/cuda:13.3.1-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    openssh-client \
    build-essential \
    numactl \
    time \
    cmake \
    unzip \
    libomp-dev \
    libboost-program-options-dev libboost-iostreams-dev libzstd-dev zlib1g-dev \
    python3 python3-pip python3-venv python3-dev python-is-python3 \
    gpg-agent wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=ssh \
    git clone git@github.com:CornellHPC/mikado.git /opt/mikado \
    && cd /opt/mikado \
    && git checkout 054a0af \
    && git submodule update --init --recursive

# plink2 (official AVX2 glibc build) -> /usr/local/bin (on PATH). The base is
# Ubuntu/x86_64, so this dynamically-linked AVX2 build runs as-is.
RUN wget -O /tmp/plink2.zip https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_avx2_20260504.zip \
    && unzip -d /tmp/plink2 /tmp/plink2.zip \
    && install -m 0755 /tmp/plink2/plink2 /usr/local/bin/plink2 \
    && rm -rf /tmp/plink2 /tmp/plink2.zip \
    && plink2 --version

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor \
      | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y intel-oneapi-mkl-devel \
    && echo "/opt/intel/oneapi/mkl/latest/lib" | tee /etc/ld.so.conf.d/mkl.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# BOLT-LMM v2.5 built from source -> /opt/BOLT-LMM_v2.5/bolt. The precompiled binary
# segfaults on modern glibc (its __wrap_memcpy compat shim recurses before main), so we
# build it ourselves. The patch and the rationale for each edit live in
# /opt/mikado/code/bolt_patch (see its README.md); build_bolt.sh downloads the upstream
# tarball, applies the patch, builds with g++ + threaded MKL, installs the binary, and
# preflights it by running `bolt -h`.
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
RUN /opt/mikado/code/bolt_patch/build_bolt.sh

RUN pip install --break-system-packages \
    setuptools \
    pytest

RUN cd /opt/mikado/grgl \
    && git submodule update --init --recursive \
    && python setup.py bdist_wheel \
    && pip install --break-system-packages --force-reinstall dist/*.whl

RUN cd /opt/mikado/grapp \
    && pip install --break-system-packages .

RUN cd /opt/mikado/grg-spmv \
    && pip install --break-system-packages .[gpu]

RUN cd /opt/mikado/code/trsv \
    && make \
    && install -m 0755 sptrsv /usr/local/bin/sptrsv \
    && sptrsv --help
