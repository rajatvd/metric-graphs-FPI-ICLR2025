CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
ARCH = sm_86  # Change to match your GPU
PTX_PATH = lib/kernel.ptx
CUBIN_PATH = lib/kernel.cubin

all: $(PTX_PATH) $(CUBIN_PATH)

$(PTX_PATH): src/langevin_kernel.cu include/langevin_kernel.h
	@mkdir -p lib
	$(NVCC) -ptx -arch=$(ARCH) -Iinclude -o $@ $<

$(CUBIN_PATH): src/langevin_kernel.cu include/langevin_kernel.h
	$(NVCC) -cubin -arch=$(ARCH) -Iinclude -o $@ $<

clean:
	rm -rf lib
