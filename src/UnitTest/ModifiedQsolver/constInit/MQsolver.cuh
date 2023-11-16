#ifndef __MQSOLVER_CUH
#define __MQSOLVER_CUH

#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Stream/StreamfuncModified.cuh>
#include <TimeIntegration/RK4.cuh>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <BasicUtils/UtilFuncs.hpp>

// to denote the type of initialization
enum InitType{
    Phy_init,
    Spec_init,
    File_init
};
 // preclaimation
void r1p_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);

void r2p_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);

void wp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);

void r1sp_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);

void r2sp_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);

void wsp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);

void precompute_func(Field* r1, Field* r2, Field* w, InitType flag);

// definitions
void file_init(string filename, Field* f){
    int Nx = f->mesh->Nx;
    int Ny = f->mesh->Ny;

    ifstream infile(filename);
    vector<vector<string>> data;
    string strline;
    while(getline(infile, strline)){
        stringstream ss(strline);
        string str;
        vector<string> dataline;

        while(getline(ss, str, ',')){
            dataline.push_back(str);
        }
        data.push_back(dataline);
    }

    if(data.size() != Ny || data[0].size() != Nx){
        printf("READ_CSV_ERROR: size not match! \n");
        return;
    }
    int index = 0;
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            index = j*Nx + i;
            f->phys[index] = string2num<double>(data[j][i]);
        }
    }
    infile.close();
}

void r1p_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            float x = dx*i;
            float y = dy*j;
            r1[index] = 0.001 + 0.0001*(double(rand())/RAND_MAX-0.5);
        }
    }
}

void r2p_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            float x = dx*i;
            float y = dy*j;
            r2[index] = 0.001 + 0.0001*(double(rand())/RAND_MAX-0.5);
        }
    }
}

void wp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            w[index] = 0.0000;
        }
    }
}

void Ra_init(Qreal *Ra, Qreal ra, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            Ra[index] = (ra);
        }
    }
}

void precompute_func(Field* r1, Field* r2, Field* w, InitType flag){
    Mesh* mesh = r1->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny;
    Qreal dx = mesh->dx; Qreal dy = mesh->dy;

    if (flag == Phy_init){
        r1p_init(r1->phys, dx, dy, Nx, Ny);
        r2p_init(r2->phys, dx, dy, Nx, Ny);
        wp_init(w->phys, dx, dy, Nx, Ny);
    }
    else if (flag == File_init){
        file_init("./init/r1_init.csv", r1);
        file_init("./init/r2_init.csv", r2);
        file_init("./init/w_init.csv", w);
    }

    FwdTrans(mesh, r1->phys, r1->spec);
    FwdTrans(mesh, r2->phys, r2->spec);
    FwdTrans(mesh, w->phys, w->spec);
}
#endif