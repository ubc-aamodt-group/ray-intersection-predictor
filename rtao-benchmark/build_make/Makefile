PROJECT_NAME = CWBVH
PROJECT_NAME_MAGIC = magic_CWBVH
BUILD_DIR = ./
SRC_DIR = ../src/
INC_DIR = ../include/

# Embree3
EMBREE_INCLUDE_DIR = $(HOME)/downloads/embree-3.13.0.x86_64.linux/include
EMBREE_LIB = $(HOME)/downloads/embree-3.13.0.x86_64.linux/lib

# tbb
TBB_INCLUDE_DIR = $(HOME)/intel/oneapi/tbb/2021.2.0/include
TBB_LIB = $(HOME)/intel/oneapi/tbb/2021.2.0/lib/intel64/gcc4.8

# Other flags
profile=0
debug=0


# CUDA
# plain cuda, use this option for running on hardware not on gpgpu-sim
# CUDAPATH = ~/cuda/include
NVCC = $(CUDA_HOME)/bin/nvcc
NVCCFLAGS = -c -g -std=c++11
CUDAARCH = -gencode arch=compute_75,code=compute_75

LFLAGS = -lcuda --cudart shared -lembree3 -ltbb
DFLAGS =

ifeq ($(profile), 1)
		DFLAGS += -DENABLE_PROFILING
endif

ifeq ($(debug), 1)
		GPUDEBUGFLAG = -g -G # Only use it when debugging the kernel in cuda gdb, don't set this when running with GPGPU-Sim!
		DFLAGS += -DDEBUG
else
		GPUDEBUGFLAG =
endif


all: regular magic

magic: gpu_magic cpu_magic stb_image
		$(NVCC) $(GPUDEBUGFLAG) $(CUDAARCH) -o $(BUILD_DIR)/$(PROJECT_NAME_MAGIC) *.o -L $(EMBREE_LIB) -L $(TBB_LIB) $(LFLAGS)

regular: gpu cpu stb_image
		$(NVCC) $(GPUDEBUGFLAG) $(CUDAARCH) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o -L $(EMBREE_LIB) -L $(TBB_LIB) $(LFLAGS)

gpu:
		$(NVCC) --keep $(NVCCFLAGS) $(CUDAARCH) $(SRC_DIR)*.cu $(DFLAGS)

cpu:
		$(NVCC) $(NVCCFLAGS) $(CUDAARCH) $(SRC_DIR)*.cpp -I $(INC_DIR) -I $(EMBREE_INCLUDE_DIR) -I $(TBB_INCLUDE_DIR) $(DFLAGS)

gpu_magic:
		$(NVCC) --keep $(NVCCFLAGS) $(CUDAARCH) $(SRC_DIR)*.cu $(DFLAGS) -DMAGIC

cpu_magic:
		$(NVCC) $(NVCCFLAGS) $(CUDAARCH) $(SRC_DIR)*.cpp -I $(INC_DIR) -I $(EMBREE_INCLUDE_DIR) -I $(TBB_INCLUDE_DIR) $(DFLAGS) -DMAGIC

stb_image:
		$(NVCC) $(NVCCFLAGS) $(CUDAARCH) $(INC_DIR)stb_image/stb_image.cpp $(DFLAGS)

clean:
		find . -type f -name _cuobjdump_list_ptx_\* -exec rm {} \;
		find . -type f -name _app_cuda_version_\* -exec rm {} \;
		rm -f *.o

help:
		@echo ""
		@echo "Options:"
		@echo "    > all: generates regular and 'magic' version"
		@echo "    > clean: clean up files"
		@echo "    use_optix=1 to use OptiX"
		@echo "    profile=1 to enable profiling (not for use with gpgpu-sim)"
		@echo ""
