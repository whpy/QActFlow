#include <BasicUtils/UtilFuncs.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>

int main(){
    int BSZ = 4;
    int Nx = 8;
    int Ny = 8;
    int Lx = 2*M_PI;
    int Ly = 2*M_PI;
    Mesh *mesh1 = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Mesh *mesh2 = new Mesh(BSZ, Nx/2, Ny/2, Lx, Ly);
    Field *fld = new Field(mesh1);
    Field *ifld = new Field(mesh1);
    Field *bfld = new Field(mesh2); 
    FldSet<<<mesh1->dimGridp,mesh1->dimBlockp>>>(fld->phys, 0.258, Nx, Ny, BSZ);

    cuda_error_func( cudaDeviceSynchronize());
    field_visual(fld, "test.csv");
    field_visual(ifld, "b4itest.csv");
    field_visual(ifld, "bdtest.csv");
    file_init("test.csv", ifld);
    field_visual(ifld, "itest.csv");
    file_init("test.csv", bfld);
    return 0;
}