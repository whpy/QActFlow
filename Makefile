NVCC = nvcc
ROOT = ../../..
INC = $(ROOT)/src
SRC = $(ROOT)/src
all: build

.PHONY: build
build:
	$(NVCC) -I $(INC) -c $(SRC)/Field/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/TimeIntegration/RK4.cu
	$(NVCC) -I $(INC) -c main.cu
	$(NVCC) -o main main.o FldOp.o Mesh.o Field.o RK4.o -lcufft
#	rm *.csv *.o

.PHONY: clean
clean:
	rm *.o main *.csv
