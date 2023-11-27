ROOT = ../..
include $(ROOT)/../config_flag
INC = -I $(ROOT) -I /usr/local/cuda/include/
SRC = $(ROOT)
LIBRARIES := -lcufft

all: build

.PHONY: build
build: 
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/Field.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Basic/FldOp.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Stream/Streamfunc_dec.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/TimeIntegration/RK4.cu

Qsolver:FldOp.o Field.o Mesh.o RK4.o Streamfunc_dec.o
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c Qsolver.cu
	$(NVCC) -o Qsolver.out Qsolver.o FldOp.o Field.o Mesh.o RK4.o Streamfunc_dec.o -lcufft

.PHONY: clean
clean:
	rm main *.csv *.out
