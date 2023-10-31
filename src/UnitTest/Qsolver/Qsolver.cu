#include "Qsolver.cuh"

void r1_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            r1[index] = rand()*1.0/RAND_MAX - 0.5;
        }
    }
}

void r2_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            r2[index] = rand()*1.0/RAND_MAX - 0.5;
        }
    }
}


void w_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny){
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = i+j*Nx;
            w[index] = 0.0;
        }
    }
}

void precompute_func(Field* r1, Field* r2, Field* w){
    Mesh* mesh = r1->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    Qreal dx = mesh->dx; Qreal dy = mesh->dy;

    r1_init(r1->phys, dx, dy, Nx, Ny);
    r2_init(r2->phys, dx, dy, Nx, Ny);
    w_init(w->phys, dx, dy, Nx, Ny);

    FwdTrans(mesh, r1->phys, r1->spec);
    FwdTrans(mesh, r2->phys, r2->spec);
    FwdTrans(mesh, w->phys, w->spec);
}

int main(){
    // computation parameters
    int BSZ = 16;
    int Ns = 4000;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 2*M_PI;
    Qreal Ly = 2*M_PI;
    Qreal dx = 2*M_PI/Nx;
    Qreal dy = 2*M_PI/Ny;
    Qreal dt = 0.002; // same as colin
    Qreal a = 1.0;

    //////////////////////// variables definitions //////////////////////////
    // parameters
    Qreal lambda;
    Qreal cn;
    Qreal Er;
    Qreal Re;
    Qreal Pe;
    Qreal cf ;

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *w_old = new Field(mesh); Field *w_curr = new Field(mesh); Field *w_new = new Field(mesh);
    Field *r1_old = new Field(mesh); Field *r1_curr = new Field(mesh); Field *r1_new = new Field(mesh);
    Field *r2_old = new Field(mesh); Field *r2_curr = new Field(mesh); Field *r2_new = new Field(mesh);

    precompute_func(r1_old, r2_old, w_old);
    BwdTrans(mesh, r1_old->spec, r1_old->phys);
    BwdTrans(mesh, r2_old->spec, r2_old->phys);
    BwdTrans(mesh, w_old->spec, w_old->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    coord(*mesh);
    field_visual(r1_old, "r1.csv");
    field_visual(r2_old, "r2.csv");
    field_visual(w_old, "w.csv");
}