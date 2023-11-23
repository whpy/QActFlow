ROOT = ../../
include $(ROOT)/../config_flag
INC = -I $(ROOT) -I /usr/local/cuda/include/
SRC = $(ROOT)
LIBRARIES := -lcufft

# HOST_COMPILER := gcc
# NVCC          := $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)

# GENCODE_FLAGS := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
# ALL_CCFLAGS := -m64    --threads 0 --std=c++11



nonlterms:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(SRC)/Basic/Field.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(SRC)/Basic/Mesh.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(SRC)/TimeIntegration/RK4.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c nonlterms.cu
	$(NVCC) -o main nonlterms.o Field.o Mesh.o RK4.o -lcufft
	

BasicOp:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c BasicOp.cu
	$(NVCC) -o main BasicOp.o -lcufft
#	rm *.csv *.o

BFwd:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/Field.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c BFwd.cu
	$(NVCC) -o BFwd.out BFwd.o Field.o Mesh.o -lcufft

deriv:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/Field.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/FldOp.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Stream/StreamfuncModified.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c deriv.cu
	$(NVCC) -o deriv.out deriv.o Field.o Mesh.o FldOp.o StreamfuncModified.o -lcufft


.PHONY: clean
clean:
	rm *.o main *.csv *.out
