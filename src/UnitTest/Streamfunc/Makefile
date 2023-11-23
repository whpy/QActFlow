ROOT = ../../
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
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $(SRC)/Stream/Streamfunc.cu

nonlterms:Field.o Mesh.o FldOp.o Streamfunc.o
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c nonlterms.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o nonlterms.out nonlterms.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms2:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c nonlterms2.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o nonlterms2.out nonlterms2.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms3:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c nonlterms3.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o nonlterms3.out nonlterms3.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms4:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c nonlterms4.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o nonlterms4.out nonlterms4.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms5:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c nonlterms5.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o nonlterms5.out nonlterms5.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms6:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c nonlterms6.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o nonlterms6.out nonlterms6.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

crossterm:
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c crossterm.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o crossterm.out crossterm.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft
.PHONY: clean
clean:
	rm main *.csv *.out
