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
    python3 python3-pip python3-venv python3-dev python-is-python3 \
    gpg-agent wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages \
    numpy \
    scipy \
    tqdm \
    cupy-cuda13x \
    pytest \
    matplotlib \
    torch

RUN mkdir -p /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN --mount=type=ssh \
    git clone git@github.com:CornellHPC/grgl.git /opt/grgl \
    && cd /opt/grgl \
    && git checkout 5a86678 \
    && git submodule update --init --recursive \
    && GRGL_CUDA=1 pip install --break-system-packages -e .

RUN --mount=type=ssh \
    git clone git@github.com:CornellHPC/mikado.git /opt/mikado \
    && cd /opt/mikado \
    && git checkout dfb659a

