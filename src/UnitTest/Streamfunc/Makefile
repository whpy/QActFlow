NVCC = nvcc
ROOT = ../../..
INC = $(ROOT)/src
SRC = $(ROOT)/src

all: build

.PHONY: build
build: 
	

nonlterms:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms.cu
	$(NVCC) -o nonlterms.out nonlterms.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms2:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms2.cu
	$(NVCC) -o nonlterms2.out nonlterms2.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms3:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms3.cu
	$(NVCC) -o nonlterms3.out nonlterms3.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms4:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms4.cu
	$(NVCC) -o nonlterms4.out nonlterms4.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

nonlterms5:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms5.cu
	$(NVCC) -o nonlterms5.out nonlterms5.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft
nonlterms6:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c nonlterms6.cu
	$(NVCC) -o nonlterms6.out nonlterms6.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft
crossterm:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/Streamfunc.cu
	$(NVCC) -I $(INC) -c crossterm.cu
	$(NVCC) -o main crossterm.o Field.o Mesh.o FldOp.o Streamfunc.o -lcufft

.PHONY: clean
clean:
	rm *.o main *.csv *.out
