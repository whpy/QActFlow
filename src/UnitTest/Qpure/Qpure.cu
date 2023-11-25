#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Basic/cuComplexBinOp.hpp>
#include <Stream/Streamfunc.cuh>
#include <TimeIntegration/RK4.cuh>

#include <stdlib.h>
#include <iostream>
#include <random>

using namespace std;

// the classical taylor green vortex

void init_func(Qreal* r1, Qreal* r2, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ){
    for (int i=0; i<Nx; i++){
        for(int j=0; j<Ny; j++){
            int index = i + j*Nx;
            Qreal x = i*dx;
            Qreal y = j*dy;
            // fp[index] = exp(-1.0* ((x-dx*Nx/2)*(x-dx*Nx/2) + (y-dy*Ny/2)*(y-dy*Ny/2)) );
            // fp[index] = -2*cos(x)*cos(y);
            // r1[index] = 0.2*(sin(x)*sin(x) - 0.5);
            // r2[index] = 0.2*sin(x)*cos(x);
            r1[index] = 2*(Qreal(rand())/RAND_MAX);
            r2[index] = 2*(Qreal(rand())/RAND_MAX);
        }
    }
}

void field_visual(Field *f, string name){
    Mesh* mesh = f->mesh;
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<mesh->Ny; j++){
        for (int i=0; i<mesh->Nx; i++){
            fval << f->phys[j*mesh->Nx+i] << ",";
        }
        fval << endl;
    }
    fval.close();
}

__global__
void r1lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Re,
Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    Qreal alpha = -1.0*(k_squared[index]) + 1;
    if(i<Nxh && j<Ny){
        IFr1h[index] = exp( alpha *dt/2);
        IFr1[index] = exp( alpha *dt);
    }
}

__global__
void r2lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Re,
Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    Qreal alpha = -1.0*(k_squared[index]) + 1;
    if(i<Nxh && j<Ny){
        IFr1h[index] = exp( alpha *dt/2);
        IFr1[index] = exp( alpha *dt);
    }
}

void r1nonl_func(Field* r1nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
    Mesh* mesh = r1nonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    BwdTrans(mesh, r1curr->spec, r1curr->phys);
    BwdTrans(mesh, r2curr->spec, r2curr->phys);
    // nonl = -S^2 r1
    //cal the S
    S_func(r1curr, r2curr, S);

    FldMul<<<dimGrid, dimBlock>>>(S->phys,S->phys,-1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r1nonl->phys,r1curr->phys,1.0,r1nonl->phys,mesh->Nx, mesh->Ny, mesh->BSZ);

    FwdTrans(mesh, r1nonl->phys, r1nonl->spec);
    
}

void r2nonl_func(Field* r2nonl, Field* r1curr, Field* r2curr, Field* S, Qreal t, Field *aux){
    Mesh* mesh = r2nonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    BwdTrans(mesh, r1curr->spec, r1curr->phys);
    BwdTrans(mesh, r2curr->spec, r2curr->phys);
    // nonl = -S^2 r2
    //cal the S
    S_func(r1curr, r2curr, S);

    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FldMul<<<dimGrid, dimBlock>>>(r2nonl->phys, r1curr->phys, 1.0, r2nonl->phys, mesh->Nx, mesh->Ny, mesh->BSZ);

    FwdTrans(mesh, r2nonl->phys, r2nonl->spec);
    
}

void print_spec(Field* f){
    Mesh* mesh = f->mesh;
    int Nxh = mesh->Nxh, Ny = mesh->Ny;
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = i + j*Nxh;
            cout << "("<< f->spec[index].x << "," << f->spec[index].y << ")" << " ";
        }
        cout << endl;
    }
}

void print_phys(Field* f){
    Mesh* mesh = f->mesh;
    int Nx = mesh->Nx, Ny = mesh->Ny;
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            cout << "("<< f->phys[index]<< ")" << " ";
        }
        cout << endl;
    }
}

void coord(Mesh &mesh){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<mesh.Ny; j++){
        for ( int i=0; i< mesh.Nx; i++){
            double x = mesh.dx*i;
            double y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}
// we test the performance of the RK4 on linear ODE that du/dt = -u where
// the exact solution should be u = c0*exp(-t), c0 depends on initial conditon.
int main(){
    int BSZ = 16;
    int Ns = 5001;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 4*M_PI;
    Qreal Ly = 4*M_PI;
    Qreal dx = Lx/Nx;
    Qreal dy = Ly/Ny;
    Qreal dt = 0.01; // same as colin
    // Qreal a = 1.0;

    Qreal Re = 0.1;

    // Fldset test
    // vortex fields
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *r1old = new Field(mesh); 
    Field *r1curr = new Field(mesh); 
    Field *r1new = new Field(mesh);
    Field *r1nonl = new Field(mesh);

    Field *r2old = new Field(mesh); 
    Field *r2curr = new Field(mesh); 
    Field *r2new = new Field(mesh);
    Field *r2nonl = new Field(mesh);

    Field *S = new Field(mesh);
    
    // time integrating factors
    Qreal *IFr1, *IFr1h;
    cudaMallocManaged(&IFr1, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFr1h, sizeof(Qreal)*Nxh*Ny);

    Qreal *IFr2, *IFr2h;
    cudaMallocManaged(&IFr2, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFr2h, sizeof(Qreal)*Nxh*Ny);

    // auxiliary field
    Field *aux = new Field(mesh);

    coord(*mesh);
    // int m = 0;
    // initialize the field
    // set up the Integrating factor
    // we may take place here by IF class
    r1lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr1h, IFr1, mesh->k_squared, Re, 
    dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFr2h, IFr2, mesh->k_squared, Re, 
    dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // initialize the physical space of Q
    init_func(r1old->phys, r2old->phys, mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    // initialize the spectral space of w 
    FwdTrans(mesh, r1old->phys, r1old->spec);
    FwdTrans(mesh, r2old->phys, r2old->spec);

    BwdTrans(mesh, r1old->spec, r1old->phys);
    BwdTrans(mesh, r2old->spec, r2old->phys);
    S_func(r1old, r2old, S);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(r1old, "00r1.csv");
    field_visual(r2old, "00r2.csv");
    field_visual(S, "00S.csv");
    field_visual(r1nonl, "00r1nonl.csv");
    field_visual(r2nonl, "00r2nonl.csv");
    for(int m=0 ;m<Ns ;m++){
        integrate_func0(r1old, r1curr, r1new, IFr1, IFr1h, dt);
        integrate_func0(r2old, r2curr, r1new, IFr2, IFr2h, dt);
        
        r1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        r2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func1(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        integrate_func1(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        
        r1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        r2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func2(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        integrate_func2(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        
        r1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        r2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func3(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        integrate_func3(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        
        r1nonl_func(r1nonl, r1curr, r2curr, S, m*dt, aux);
        r2nonl_func(r2nonl, r1curr, r2curr, S, m*dt, aux);
        cuda_error_func( cudaDeviceSynchronize() );
        integrate_func4(r1old, r1curr, r1new, r1nonl, IFr1, IFr1h, dt);
        integrate_func4(r2old, r2curr, r2new, r2nonl, IFr2, IFr2h, dt);
        
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

        if(m%1 == 0) cout << "t = " << m*dt << endl;
        if (m%20 == 0){
            BwdTrans(mesh, r1old->spec, r1old->phys);
            BwdTrans(mesh, r2old->spec, r2old->phys);
            S_func(r1old, r2old, S);
            cuda_error_func( cudaDeviceSynchronize() );
            field_visual(r1old, to_string(m)+"r1.csv");
            field_visual(r2old, to_string(m)+"r2.csv");
            field_visual(r1nonl, to_string(m)+"r1nonl.csv");
            field_visual(r2nonl, to_string(m)+"r2nonl.csv");
            field_visual(S, to_string(m)+"S.csv");
            // printf("t: %f    val:%.8f   exa:%.8f    err: %.8f \n",m*dt,  u->phys[5],exact((m)*dt), u->phys[5]-exact((m)*dt));
            // cout<<"t: " << m*dt << "  " << u->phys[5] << endl;
        }
    }

    delete mesh;
    
    return 0;
}
