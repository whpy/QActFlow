#ifndef __QSOLVER_CUH
#define __QSOLVER_CUH

#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
// #include <Stream/Streamfunc_dec.cuh>
#include <Stream/Streamfunc.cuh>
#include <TimeIntegration/RK4.cuh>

#include <stdlib.h>
#include <time.h>
#include <BasicUtils/UtilFuncs.hpp>

// to denote the type of initialization
enum eInitType{
    Phy_init,
    Spec_init,
    File_init
};

string InitType[] = {
    "Physical init",
    "Spectral init",
    "File init"
};
 // preclaimation
void r1p_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);

void r2p_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);

void wp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);

void r1sp_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);

void r2sp_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);

void wsp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);

void precompute_func(Field* r1, Field* r2, Field* w, eInitType flag);

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
            // r1[index] = (double(rand())/RAND_MAX-1)*0.5;
            r1[index] = 0.5*(sin((x+y))*sin((x+y)) - 0.5);
        }
    }
}

void r2p_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            float x = dx*i;
            float y = dy*j;
            // r2[index] = (double(rand())/RAND_MAX-1)*0.5;
            r2[index] = 0.2*sin((x+y))*cos((x+y));
        }
    }
}

void wp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            float x = dx*i;
            float y = dy*j;
            // w[index] = -2*cos(x)*cos(y);
            w[index] = 0.0;
        }
    }
}

void r1sp_init(Qcomp *r1, int Nxh, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int index = i+j*Nxh;
            double theta = (float(rand())/RAND_MAX)*2*M_PI;
            double alt = ((float(rand())/RAND_MAX)-0.5)/(2*Nxh*Ny);
            r1[index] = make_cuDoubleComplex(alt*cos(theta),alt*sin(theta));
        }
    }
}

void r2sp_init(Qcomp *r2, int Nxh, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int index = i+j*Nxh;
            double theta = (float(rand())/RAND_MAX)*2*M_PI;
            double alt = ((float(rand())/RAND_MAX)-0.5)/(2*Nxh*Ny);
            r2[index] = make_cuDoubleComplex(alt*cos(theta),alt*sin(theta));
        }
    }
}

void wsp_init(Qcomp *w, int Nxh, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int index = i+j*Nxh;
            w[index] = make_cuDoubleComplex(0.000,0.000);
        }
    }
}

void alpha_init(Qreal *alpha, Qreal Ra, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            alpha[index] = (Ra);
        }
    }
}

void precompute_func(Field* r1, Field* r2, Field* w, eInitType flag){
    Mesh* mesh = r1->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny;
    Qreal dx = mesh->dx; Qreal dy = mesh->dy;

    if (flag == Phy_init){
        r1p_init(r1->phys, dx, dy, Nx, Ny);
        r2p_init(r2->phys, dx, dy, Nx, Ny);
        wp_init(w->phys, dx, dy, Nx, Ny);
        FwdTrans(mesh, r1->phys, r1->spec);
        FwdTrans(mesh, r2->phys, r2->spec);
        FwdTrans(mesh, w->phys, w->spec);
    }
    else if (flag == File_init){
        file_init("./init/r1_init.csv", r1);
        file_init("./init/r2_init.csv", r2);
        file_init("./init/w_init.csv", w);
        FwdTrans(mesh, r1->phys, r1->spec);
        FwdTrans(mesh, r2->phys, r2->spec);
        FwdTrans(mesh, w->phys, w->spec);
    }
    else if (flag == Spec_init){
        r1sp_init(r1->spec, mesh->Nxh, mesh->Ny);
        r2sp_init(r2->spec, mesh->Nxh, mesh->Ny);
        wsp_init(w->spec, mesh->Nxh, mesh->Ny);
        symmetry_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        symmetry_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        symmetry_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(w->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        BwdTrans(mesh, r1->spec, r1->phys);
        BwdTrans(mesh, r2->spec, r2->phys);
        BwdTrans(mesh, w->spec, w->phys);
    }
    
}
#endif