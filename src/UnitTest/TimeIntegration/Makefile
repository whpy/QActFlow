NVCC = nvcc
ROOT = ../../..
INC = $(ROOT)/src
SRC = $(ROOT)/src

all: build

.PHONY: build
build: 
	

nonlinear1:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/TimeIntegration/RK4.cu
	$(NVCC) -I $(INC) -c nonlinear1.cu
	$(NVCC) -o main nonlinear1.o Field.o Mesh.o RK4.o -lcufft

burgers1D:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/TimeIntegration/RK4.cu
	$(NVCC) -I $(INC) -c burgers1D.cu
	$(NVCC) -o main burgers1D.o Field.o Mesh.o RK4.o -lcufft

heat2D:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/TimeIntegration/RK4.cu
	$(NVCC) -I $(INC) -c heat2D.cu
	$(NVCC) -o heat2D.out heat2D.o FldOp.o Field.o Mesh.o RK4.o -lcufft

ModTimeInt:
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Field.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/Mesh.cpp
	$(NVCC) -I $(INC) -c $(SRC)/Basic/FldOp.cu
	$(NVCC) -I $(INC) -c $(SRC)/TimeIntegration/RK4.cu
	$(NVCC) -I $(INC) -c $(SRC)/Stream/StreamfuncModified.cu
	$(NVCC) -I $(INC) -c ModTimeInt.cu
	$(NVCC) -o ModTimeInt.out ModTimeInt.o StreamfuncModified.o FldOp.o Field.o Mesh.o RK4.o -lcufft

.PHONY: clean
clean:
	rm *.o main *.csv *.out
