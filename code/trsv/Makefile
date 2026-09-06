# Toolkit-agnostic build for the cuSPARSE SpTrSV benchmark.
#
# Picks up `nvcc` from PATH and derives CUDA_HOME from it, so the same Makefile
# builds inside the CUDA 13.3 Docker image and on the Perlmutter CUDA 12.9 host.
# The NVIDIA HPC SDK ships cuSPARSE in a separate math_libs/ tree rather than in
# cuda/<ver>/lib64, so we also search there. Everything is overridable:
#   make                       # nvcc on PATH, default sm_80
#   make ARCH=sm_90            # Hopper
#   make CUDA_HOME=/path/to/cuda
#   make CUSPARSE_DIR=/path/with/include+lib    # force cuSPARSE location
#
NVCC      ?= nvcc
# Fall back to the parent of wherever nvcc resolves to (e.g. /usr/local/cuda).
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC)))
ARCH      ?= sm_80

# Locate libcusparse.so: check the toolkit's own lib dirs first, then the HPC
# SDK math_libs tree that lives two levels up from cuda/<ver> (.../<ver>/cuda/..
# -> .../<ver>/math_libs/*/targets/*/lib). CUSPARSE_DIR overrides the search.
ifndef CUSPARSE_DIR
  _cusparse_so := $(firstword $(wildcard \
      $(CUDA_HOME)/lib64/libcusparse.so* \
      $(CUDA_HOME)/lib/libcusparse.so* \
      $(CUDA_HOME)/../../math_libs/*/targets/*/lib/libcusparse.so*))
  # Strip ".../lib/libcusparse.so*" down to the package root.
  CUSPARSE_DIR := $(patsubst %/lib/,%,$(dir $(_cusparse_so)))
endif

CUSPARSE_INC := $(firstword $(wildcard $(CUSPARSE_DIR)/include $(CUDA_HOME)/include))
CUSPARSE_LIB := $(firstword $(wildcard $(CUSPARSE_DIR)/lib $(CUDA_HOME)/lib64))

INCFLAGS  := $(if $(CUDA_HOME),-I$(CUDA_HOME)/include,) $(if $(CUSPARSE_INC),-I$(CUSPARSE_INC),)
LIBFLAGS  := $(if $(CUDA_HOME),-L$(CUDA_HOME)/lib64,) $(if $(CUSPARSE_LIB),-L$(CUSPARSE_LIB),)
# Embed the cuSPARSE lib dir as an rpath so no LD_LIBRARY_PATH is needed at runtime.
RPATHFLAG := $(if $(CUSPARSE_LIB),-Xlinker -rpath=$(CUSPARSE_LIB),)

sptrsv: sptrsv.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $(INCFLAGS) $< -o $@ \
	    $(LIBFLAGS) $(RPATHFLAG) -lcusparse -lcudart

clean:
	rm -f sptrsv

.PHONY: clean
